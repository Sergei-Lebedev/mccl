#include "config.h"
#include "xccl_ucx_lib.h"
#include "alltoall.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

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

#define GET_RANK_FROM_TAG(_tag) \
    ((uint32_t)(((TEAM_UCX_RANK_MASK) & (_tag)) >> TEAM_UCX_RANK_BITS_OFFSET))


enum {
    SLOT_READY     = 0x0ul,
    SLOT_NOT_READY = 0x1ul
};

static inline xccl_status_t copy_req_testany(xccl_mem_component_request_t **reqs,
                                             int n_reqs, int *completed_idx,
                                             uint64_t *tag)
{
    int i;
    xccl_status_t st;

    assert(NULL != reqs);
    for (i=0; i<n_reqs; i++) {
        if (SLOT_NOT_READY == (uint64_t)reqs[i]) {
            continue;
        }

        if (SLOT_READY == reqs[i]) {
            *completed_idx = i;
            reqs[i] = (void*)SLOT_NOT_READY;
            return XCCL_OK;
        } else {
            st = xccl_mem_component_test_request(reqs[i]);
            if (st == XCCL_OK) {
                *tag = reqs[i]->tag;
                *completed_idx = i;
                xccl_mem_component_free_request(reqs[i]);
                reqs[i] = (void*)SLOT_NOT_READY;
                return XCCL_OK;
            }
        }
    }
    return XCCL_INPROGRESS;
}

static inline xccl_status_t ucx_req_testany(xccl_ucx_team_t *team,
                                            xccl_ucx_request_t **reqs,
                                            int n_reqs, int *completed_idx,
                                            uint64_t *tag)
{
    int i;

    assert(NULL != reqs);
    for (i=0; i<n_reqs; i++) {
        if (SLOT_NOT_READY == (uint64_t)reqs[i]) {
            continue;
        }

        if (SLOT_READY == reqs[i]) {
            *completed_idx = i;
            *tag = 1234;
            reqs[i] = (void*)SLOT_NOT_READY;
            return XCCL_OK;
        } else {
            if (reqs[i]->status != XCCL_UCX_REQUEST_DONE) {
                xccl_ucx_progress(team);
            } else {
                *tag = (uint64_t)reqs[i]->sender_tag;
                *completed_idx = i;
                xccl_ucx_req_free(reqs[i]);
                reqs[i] = (void*)SLOT_NOT_READY;
                return XCCL_OK;
            }
        }
    }
    return XCCL_INPROGRESS;
}

static inline xccl_status_t copy_req_testall(xccl_mem_component_request_t **reqs,
                                             int n_reqs)
{
    int i;
    xccl_status_t st;

    assert(NULL != reqs);
    for (i=0; i<n_reqs; i++) {
        if (((uint64_t)reqs[i] != SLOT_NOT_READY) && ((uint64_t)reqs[i] != SLOT_READY)) {
            st = xccl_mem_component_test_request(reqs[i]);
            if (st == XCCL_OK) {
                xccl_mem_component_free_request(reqs[i]);
                reqs[i] = (void*)SLOT_NOT_READY;
            } else {
                return XCCL_INPROGRESS;
            }
        }
    }

    return XCCL_OK;
}

static inline xccl_status_t ucx_req_testall(xccl_ucx_team_t *team,
                                            xccl_ucx_request_t **reqs,
                                            int n_reqs)
{
    int i;
    xccl_status_t st = XCCL_OK;

    assert(NULL != reqs);
    for (i=0; i<n_reqs; i++) {
        if (((uint64_t)reqs[i] != SLOT_NOT_READY) && ((uint64_t)reqs[i] != SLOT_READY)) {
            if (reqs[i]->status != XCCL_UCX_REQUEST_DONE) {
                xccl_ucx_progress(team);
                st = XCCL_INPROGRESS;
            } else {
                xccl_ucx_req_free(reqs[i]);
                reqs[i] = (void*)SLOT_NOT_READY;
            }
        }
    }

    return st;
}

xccl_status_t xccl_ucx_alltoall_bcopy_progress(xccl_ucx_collreq_t *req)
{
    ptrdiff_t          sbuf       = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t          rbuf       = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t    *team       = ucs_derived_of(req->team, xccl_ucx_team_t);
    size_t             data_size   = req->args.buffer_info.len;
    int                group_rank  = team->super.params.oob.rank;
    int                group_size  = team->super.params.oob.size;
    xccl_ucx_request_t **reqs      = req->alltoall_bcopy.reqs;
    int                chunk       = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int                reverse     = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int                max_polls   = TEAM_UCX_CTX(team)->num_to_probe;
    ptrdiff_t          scratch_src = (ptrdiff_t)req->alltoall_bcopy.scratch_src;
    ptrdiff_t          scratch_dst = (ptrdiff_t)req->alltoall_bcopy.scratch_dst;
    int                total_reqs  = (chunk > group_size - 1 || chunk <= 0) ?
                                      group_size - 1: chunk;
    int step, peer, released_slot, n_polls;
    xccl_mem_component_request_t **copy_reqs;
    uint64_t tag;
    xccl_status_t status;

    n_polls = 0;
    copy_reqs = req->alltoall_bcopy.copy_reqs;
    while (n_polls++ < max_polls &&
           (req->alltoall_bcopy.n_copy_rreqs != group_size ||
            req->alltoall_bcopy.n_sreqs != group_size)) {
        if (req->alltoall_bcopy.n_copy_rreqs < group_size - 1) {
            status = ucx_req_testany(team, reqs, total_reqs, &released_slot, &tag);
            if (XCCL_OK == status) {
                uint32_t send_rank = GET_RANK_FROM_TAG(tag);
                xccl_mem_component_copy_async((void*)(rbuf + send_rank*data_size),
                                              req->dst_mem_type,
                                              (void*)(scratch_dst + released_slot*data_size),
                                              req->dst_mem_type,
                                              data_size,
                                              &(req->alltoall_bcopy.copy_reqs[released_slot]));
                n_polls = 0;
                req->alltoall_bcopy.n_copy_rreqs++;
            }
        }

        if (req->alltoall_bcopy.n_rreqs < group_size - 1) {
            status = copy_req_testany(copy_reqs, total_reqs, &released_slot, &tag);
            if (XCCL_OK == status) {
                peer = get_recv_peer(group_rank, group_size,
                                     req->alltoall_bcopy.n_rreqs, reverse);
                xccl_ucx_recv_nb((void*)(scratch_dst + released_slot*data_size),
                                 data_size, peer, team, req->tag, &reqs[released_slot]);
                req->alltoall_bcopy.n_rreqs++;
                n_polls = 0;
            }
        }

        if (req->alltoall_bcopy.n_copy_sreqs < group_size - 1) {
            status = ucx_req_testany(team, reqs+total_reqs, total_reqs,
                                     &released_slot, &tag);
            if (XCCL_OK == status) {
                peer = get_send_peer(group_rank, group_size,
                                     req->alltoall_bcopy.n_copy_sreqs, reverse);
                xccl_mem_component_copy_async((void*)(scratch_src + released_slot*data_size),
                                              req->src_mem_type,
                                              (void*)(sbuf + peer*data_size),
                                              req->src_mem_type,
                                              data_size,
                                              &(copy_reqs[total_reqs+released_slot]));
                copy_reqs[total_reqs+released_slot]->tag = peer;
                n_polls = 0;
                req->alltoall_bcopy.n_copy_sreqs++;
            }

        }
        if (req->alltoall_bcopy.n_sreqs < group_size - 1) {
            status = copy_req_testany(copy_reqs+total_reqs, total_reqs, &released_slot, &tag);
            if (XCCL_OK == status) {
                peer = (int)tag;
                xccl_ucx_send_nb((void*)(scratch_src + released_slot*data_size),
                                 data_size, peer, team, req->tag,
                                 &reqs[total_reqs+released_slot]);
                req->alltoall_bcopy.n_sreqs++;
                n_polls = 0;
            }
        }
    }
    if ((req->alltoall_bcopy.n_sreqs != group_size - 1) ||
        (req->alltoall_bcopy.n_copy_rreqs != group_size - 1)) {
        return XCCL_OK;
    }

    if (XCCL_INPROGRESS == ucx_req_testall(team, reqs, 2*total_reqs)) {
        return XCCL_OK;
    }
    if (XCCL_INPROGRESS == copy_req_testall(copy_reqs, 2*total_reqs)) {
        return XCCL_OK;
    }
    if (XCCL_INPROGRESS == xccl_mem_component_test_request(req->alltoall_bcopy.self_copy)) {
        return XCCL_OK;
    }

    xccl_mem_component_free_request(req->alltoall_bcopy.self_copy);
    xccl_mem_component_free((void*)scratch_src, req->src_mem_type);
    if (req->src_mem_type != req->dst_mem_type) {
        xccl_mem_component_free((void*)scratch_dst, req->dst_mem_type);
    }
    free(reqs);
    free(copy_reqs);
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_alltoall_bcopy_start(xccl_ucx_collreq_t *req)
{
    size_t data_size      = req->args.buffer_info.len;
    int    group_rank     = req->team->params.oob.rank;
    int    group_size     = req->team->params.oob.size;
    int    chunk          = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int    reverse        = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int    total_reqs     = (chunk > group_size - 1 || chunk <= 0) ?
                            group_size - 1 : chunk;
    ptrdiff_t sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    xccl_ucx_request_t **reqs;
    xccl_status_t st;
    void *scratch_src, *scratch_dst;
    int step, peer;

    req->alltoall_bcopy.reqs = calloc(total_reqs*2,
                                      sizeof(*req->alltoall_bcopy.reqs));
    if (!req->alltoall_bcopy.reqs) {
        return XCCL_ERR_NO_MEMORY;
    }
    req->alltoall_bcopy.copy_reqs = calloc(total_reqs*2,
                                           sizeof(*req->alltoall_bcopy.copy_reqs));
    if (!req->alltoall_bcopy.copy_reqs) {
        free(req->alltoall_bcopy.reqs);
        return XCCL_ERR_NO_MEMORY;
    }

    if (req->src_mem_type == req->dst_mem_type) {
        xccl_mem_component_alloc(&scratch_src, 2*total_reqs*data_size,
                                 req->src_mem_type);
        scratch_dst = (void*)((ptrdiff_t)scratch_src + total_reqs*data_size);
    } else {
        xccl_mem_component_alloc(&scratch_src, total_reqs*data_size,
                                 req->src_mem_type);
        xccl_mem_component_alloc(&scratch_dst, total_reqs*data_size,
                                 req->dst_mem_type);
    }

    reqs = req->alltoall_bcopy.reqs;
    req->progress = xccl_ucx_alltoall_bcopy_progress;
    for (step = 0; step < total_reqs; step++) {
        peer = get_send_peer(group_rank, group_size, step, reverse);
        xccl_mem_component_copy_async((void*)((ptrdiff_t)scratch_src + step*data_size),
                                      req->src_mem_type,
                                      (void*)(sbuf + peer*data_size),
                                      req->src_mem_type,
                                      data_size,
                                      &(req->alltoall_bcopy.copy_reqs[total_reqs+step]));
        req->alltoall_bcopy.copy_reqs[total_reqs+step]->tag = peer;
        req->alltoall_bcopy.reqs[total_reqs+step] = (void*)SLOT_NOT_READY;
    }
    for (step = 0; step < total_reqs; step++) {
        peer = get_recv_peer(group_rank, group_size, step, reverse);
        xccl_ucx_recv_nb((void*)((ptrdiff_t)scratch_dst + step*data_size),
                         data_size, peer, team, req->tag, &reqs[step]);
        req->alltoall_bcopy.copy_reqs[step] = (void*)SLOT_NOT_READY;
    }
    xccl_mem_component_copy_async((void*)(rbuf + group_rank*data_size),
                                  req->dst_mem_type,
                                  (void*)(sbuf + group_rank*data_size),
                                  req->src_mem_type,
                                  data_size,
                                  &(req->alltoall_bcopy.self_copy));

    req->alltoall_bcopy.scratch_src  = scratch_src;
    req->alltoall_bcopy.scratch_dst  = scratch_dst;
    req->alltoall_bcopy.n_copy_sreqs = total_reqs;
    req->alltoall_bcopy.n_copy_rreqs = 0;
    req->alltoall_bcopy.n_rreqs      = total_reqs;
    req->alltoall_bcopy.n_sreqs      = 0;
    return req->progress(req);
}
