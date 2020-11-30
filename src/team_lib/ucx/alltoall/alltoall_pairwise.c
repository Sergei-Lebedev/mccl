#include "config.h"
#include "xccl_ucx_lib.h"
#include "alltoall.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

enum alltoall_barriers_phase {
    PHASE_SENDRECV      = 0,
    PHASE_START_BARRIER = 2,
    PHASE_TEST_BARRIER  = 3
};

static inline int get_recv_peer(int group_rank, int group_size,
                                int step, int is_reverse)
{
    if (is_reverse) {
        return (group_rank - 1 - step + group_size) % group_size;
    } else {
        return (group_rank + 1 + step) % group_size;
    }
}

static inline int get_send_peer(int group_rank, int group_size,
                                int step, int is_reverse)
{
    if (is_reverse) {
        return (group_rank + 1 + step) % group_size;
    } else {
        return (group_rank - 1 - step + group_size) % group_size;
    }
}

static inline xccl_status_t send_next_slice(xccl_ucx_collreq_t *req,
                                            int group_size,
                                            int group_rank,
                                            xccl_ucx_request_t **request)
{
    int num_slices        = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_num_slices;
    int reverse           = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);

    ptrdiff_t sbuf = (ptrdiff_t)req->args.buffer_info.src_buffer;
    int peer, step, slice, step_size, slice_size;
    xccl_status_t st;

    step_size = req->args.buffer_info.len;
    if (num_slices > step_size) {
        num_slices = 1;
    }
    step = req->alltoall_pairwise.n_sreqs;
    slice = req->alltoall_pairwise.n_sslices;
    peer = get_send_peer(group_rank, group_size, step, reverse);
    slice_size = step_size / num_slices + ((slice < step_size % num_slices)? 1 : 0);
    sbuf = sbuf + peer * step_size + req->alltoall_pairwise.sslice_offset;
    // xccl_ucx_warn("send step %d slice %d peer %d", step, slice, peer);
    st = xccl_ucx_send_nb((void*)sbuf, slice_size, peer, team, req->tag, request);
    if (st != XCCL_OK) {
        xccl_ucx_error("failed to send next slice peer %d, slice %d",
                        peer, slice);
        return st;
    }
    req->alltoall_pairwise.n_sslices =
        (req->alltoall_pairwise.n_sslices + 1) % num_slices;
    req->alltoall_pairwise.sslice_offset += slice_size;
    if (req->alltoall_pairwise.n_sslices == 0) {
        req->alltoall_pairwise.n_sreqs += 1;
        req->alltoall_pairwise.sslice_offset = 0;
    }

    return XCCL_OK;
}

static inline xccl_status_t recv_next_slice(xccl_ucx_collreq_t *req,
                                            int group_size,
                                            int group_rank,
                                            xccl_ucx_request_t **request)
{
    int num_slices        = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_num_slices;
    int reverse           = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    ptrdiff_t rbuf = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    int peer, step, slice, step_size, slice_size;
    xccl_status_t st;

    step_size = req->args.buffer_info.len;
    if (num_slices > step_size) {
        num_slices = 1;
    }
    step = req->alltoall_pairwise.n_rreqs;
    slice = req->alltoall_pairwise.n_rslices;
    peer = get_recv_peer(group_rank, group_size, step, reverse);
    slice_size = step_size / num_slices + ((slice < step_size % num_slices)? 1 : 0);
    rbuf = rbuf + peer * step_size + req->alltoall_pairwise.rslice_offset;
    // xccl_ucx_warn("recv step %d slice %d peer %d", step, slice, peer);
    st = xccl_ucx_recv_nb((void*)rbuf, slice_size, peer, team, req->tag, request);
    if (st != XCCL_OK) {
        xccl_ucx_error("failed to recv next slice peer %d, slice %d",
                        peer, slice);
        return st;
    }
    req->alltoall_pairwise.n_rslices =
        (req->alltoall_pairwise.n_rslices + 1) % num_slices;
    req->alltoall_pairwise.rslice_offset += slice_size;
    if (req->alltoall_pairwise.n_rslices == 0) {
        req->alltoall_pairwise.n_rreqs += 1;
        req->alltoall_pairwise.rslice_offset = 0;
    }

    return XCCL_OK;
}

xccl_status_t xccl_ucx_alltoall_pairwise_progress(xccl_ucx_collreq_t *req)
{
    ptrdiff_t            sbuf = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t            rbuf = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t     *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    size_t          data_size = req->args.buffer_info.len;
    int            group_rank = team->super.params.oob.rank;
    int            group_size = team->super.params.oob.size;
    xccl_ucx_request_t **reqs = req->alltoall_pairwise.reqs;
    int                 chunk = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int               reverse = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int             max_polls = TEAM_UCX_CTX(team)->num_to_probe;
    int            total_reqs = (chunk > group_size - 1 || chunk <= 0) ?
        group_size - 1 : chunk;
    int step, peer, released_slot, n_polls;
    xccl_status_t status;
    n_polls = 0;
    while (n_polls++ < max_polls &&
           (req->alltoall_pairwise.n_rreqs != group_size - 1 ||
            req->alltoall_pairwise.n_sreqs != group_size - 1)) {
        if (req->alltoall_pairwise.n_rreqs < group_size - 1) {
            status = xccl_ucx_req_test(team, reqs, total_reqs, &released_slot, 1, 1);
            if (XCCL_OK == status) {
                recv_next_slice(req, group_size, group_rank, &reqs[released_slot]);
                // peer = get_recv_peer(group_rank, group_size,
                //                      req->alltoall_pairwise.n_rreqs, reverse);
                // xccl_ucx_recv_nb((void*)(rbuf + peer * data_size),
                //                  data_size, peer, team, req->tag, &reqs[released_slot]);
                // req->alltoall_pairwise.n_rreqs++;
                n_polls = 0;
            }
        }
        if (req->alltoall_pairwise.n_sreqs < group_size - 1) {
            status = xccl_ucx_req_test(team, reqs+total_reqs, total_reqs,
                                       &released_slot, 1, 1);
            if (XCCL_OK == status) {
                send_next_slice(req, group_size, group_rank,
                                &reqs[released_slot+total_reqs]);
                // peer = get_send_peer(group_rank, group_size,
                //                      req->alltoall_pairwise.n_sreqs, reverse);
                // xccl_ucx_send_nb((void*)(sbuf + peer * data_size),
                //                  data_size, peer, team, req->tag,
                //                  &reqs[released_slot+total_reqs]);
                // req->alltoall_pairwise.n_sreqs++;
                n_polls = 0;
            }
        }
    }
    if (req->alltoall_pairwise.n_rreqs != group_size - 1 ||
        req->alltoall_pairwise.n_sreqs != group_size - 1) {
        return XCCL_OK;
    }

    if (XCCL_INPROGRESS == xccl_ucx_testall(team, reqs, 2*total_reqs)) {
        return XCCL_OK;
    }
    free(reqs);
    req->complete = XCCL_OK;
    return XCCL_OK;
}

static inline int need_barrier(int n_reqs, int group_size, int barrier_after)
{
    if ((n_reqs != group_size -1 ) && (n_reqs % barrier_after == 0)) {
        return 1;
    }
    return 0;
}

xccl_status_t xccl_ucx_alltoall_pairwise_barrier_progress(xccl_ucx_collreq_t *req)
{
    ptrdiff_t            sbuf = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t            rbuf = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t     *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    size_t          data_size = req->args.buffer_info.len;
    int            group_rank = team->super.params.oob.rank;
    int            group_size = team->super.params.oob.size;
    xccl_ucx_request_t **reqs = req->alltoall_pairwise.reqs;
    int                 chunk = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int               reverse = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int         barrier_after = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_barrier;
    int             max_polls = TEAM_UCX_CTX(team)->num_to_probe;
    int            total_reqs = (chunk > group_size - 1 || chunk <= 0) ?
        group_size - 1 : chunk;
    int step, peer, released_slot, n_polls;
    xccl_status_t status;
    n_polls = 0;

    while (n_polls++ < max_polls &&
           (req->alltoall_pairwise.n_rreqs != group_size - 1 ||
            req->alltoall_pairwise.n_sreqs != group_size - 1)) {
        if (req->alltoall_pairwise.phase == PHASE_START_BARRIER) {
            status = xccl_team_lib_ucx.super.collective_post(
                (xccl_tl_coll_req_t*)req->alltoall_pairwise.barrier_req);
            if (status != XCCL_OK) {
                xccl_ucx_error("alltoall post barrier failed");
                return status;
            }
            req->alltoall_pairwise.phase = PHASE_TEST_BARRIER;
        }
        if (req->alltoall_pairwise.phase == PHASE_TEST_BARRIER) {
            status = xccl_team_lib_ucx.super.collective_test(
                (xccl_tl_coll_req_t*)req->alltoall_pairwise.barrier_req);
            if (status == XCCL_INPROGRESS) {
                continue;
            } else if (status == XCCL_OK) {
                req->alltoall_pairwise.phase = PHASE_SENDRECV;
                req->alltoall_pairwise.ready_to_recv = 1;
                req->alltoall_pairwise.ready_to_send = 1;
                n_polls = 0;
            } else {
                xccl_ucx_error("alltoall test barrier failed");
                return status;
            }
        }
        if (req->alltoall_pairwise.n_rreqs < group_size - 1) {
            status = xccl_ucx_req_test(team, reqs, total_reqs, &released_slot, 1, 1);
            if ((XCCL_OK == status) && (req->alltoall_pairwise.ready_to_recv == 1)) {
                peer = get_recv_peer(group_rank, group_size,
                                     req->alltoall_pairwise.n_rreqs, reverse);
                xccl_ucx_recv_nb((void*)(rbuf + peer * data_size),
                                 data_size, peer, team, req->tag, &reqs[released_slot]);
                req->alltoall_pairwise.n_rreqs++;
                if (need_barrier(req->alltoall_pairwise.n_rreqs, group_size, barrier_after)) {
                    req->alltoall_pairwise.phase += 1;
                    req->alltoall_pairwise.ready_to_recv = 0;
                }
                n_polls = 0;
            }
        }
        if (req->alltoall_pairwise.n_sreqs < group_size - 1) {
            status = xccl_ucx_req_test(team, reqs+total_reqs, total_reqs,
                                       &released_slot, 1, 1);
            if ((XCCL_OK == status) && (req->alltoall_pairwise.ready_to_send == 1)) {
                peer = get_send_peer(group_rank, group_size,
                                     req->alltoall_pairwise.n_sreqs, reverse);
                xccl_ucx_send_nb((void*)(sbuf + peer * data_size),
                                 data_size, peer, team, req->tag,
                                 &reqs[released_slot+total_reqs]);
                req->alltoall_pairwise.n_sreqs++;
                if (need_barrier(req->alltoall_pairwise.n_sreqs, group_size, barrier_after)) {
                    req->alltoall_pairwise.phase += 1;
                    req->alltoall_pairwise.ready_to_send = 0;
                }
                n_polls = 0;
            }
        }
    }
    if (req->alltoall_pairwise.n_rreqs != group_size - 1 ||
        req->alltoall_pairwise.n_sreqs != group_size - 1) {
        return XCCL_OK;
    }

    if (XCCL_INPROGRESS == xccl_ucx_testall(team, reqs, 2*total_reqs)) {
        return XCCL_OK;
    }
    xccl_team_lib_ucx.super.collective_finalize(
        (xccl_tl_coll_req_t*)req->alltoall_pairwise.barrier_req);
    free(reqs);
    req->complete = XCCL_OK;
    return XCCL_OK;
}

/*
 * Common start for pairwise and pairwise with barrier. Different progress
 * functions are used depeneding on the config variable alltoall_pairwise_barrier
 */
xccl_status_t xccl_ucx_alltoall_pairwise_start(xccl_ucx_collreq_t *req)
{
    size_t data_size      = req->args.buffer_info.len;
    int    group_rank     = req->team->params.oob.rank;
    int    group_size     = req->team->params.oob.size;
    int    slices         = 2;
    int    chunk          = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int    reverse        = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int    barrier_after  = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_barrier;
    int    total_reqs     = (chunk > group_size - 1 || chunk <= 0) ?
        group_size - 1 : chunk;
    ptrdiff_t sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    xccl_ucx_request_t **reqs;
    xccl_status_t st;
    int step, peer;
    req->alltoall_pairwise.reqs = calloc(slices*total_reqs*2,
                                         sizeof(*req->alltoall_pairwise.reqs));
    if (!req->alltoall_pairwise.reqs) {
        return XCCL_ERR_NO_MEMORY;
    }
    reqs = req->alltoall_pairwise.reqs;

    req->progress = xccl_ucx_alltoall_pairwise_progress;
    if (barrier_after > 0) {
        xccl_coll_op_args_t barrier_args = {
            .coll_type       = XCCL_BARRIER,
            .alg.set_by_user = 0
        };
        st = xccl_team_lib_ucx.super.collective_init(
                &barrier_args,
                (xccl_tl_coll_req_t**)(&req->alltoall_pairwise.barrier_req),
                req->team);
        if (st != XCCL_OK) {
            xccl_ucx_error("failed to init barrier req in alltoall");
            return st;
        }
        req->progress = xccl_ucx_alltoall_pairwise_barrier_progress;
        req->alltoall_pairwise.ready_to_send = 1;
        req->alltoall_pairwise.ready_to_recv = 1;
        req->alltoall_pairwise.phase         = PHASE_SENDRECV;
        xccl_ucx_trace("alltoall with barrier will be used");
    }
    req->alltoall_pairwise.n_rreqs = 0;
    req->alltoall_pairwise.n_rslices = 0;
    req->alltoall_pairwise.rslice_offset = 0;
    req->alltoall_pairwise.n_sreqs = 0;
    req->alltoall_pairwise.n_sslices = 0;
    req->alltoall_pairwise.sslice_offset = 0;
    xccl_ucx_send_recv((void*)(sbuf+data_size*group_rank), data_size,
                       group_rank, req->tag, (void*)(rbuf+data_size*group_rank),
                       data_size, group_rank, req->tag,
                       (xccl_ucx_team_t *)req->team);
    for (step = 0; step < total_reqs; step++) {
        recv_next_slice(req, group_size, group_rank, &reqs[step]);
        // peer = get_recv_peer(group_rank, group_size, step, reverse);
        // xccl_ucx_recv_nb((void*)(rbuf + peer * data_size),
        //                  data_size, peer, team, req->tag, &reqs[step]);
    }
    for (step = 0; step < total_reqs; step++) {
        send_next_slice(req, group_size, group_rank, &reqs[step+total_reqs]);
        // peer = get_send_peer(group_rank, group_size, step, reverse);
        // xccl_ucx_send_nb((void*)(sbuf + peer * data_size),
        //                  data_size, peer, team, req->tag, &reqs[step+total_reqs]);
    }

    return req->progress(req);
}
