#include "cuda_mem_component.h"
#include <stdlib.h>
#include <assert.h>

#define NUM_REQUESTS 128

xccl_cuda_mem_component_t xccl_cuda_mem_component;

#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            return XCCL_ERR_NO_MESSAGE;                             \
        }                                                           \
} while(0)

static xccl_status_t xccl_cuda_open()
{
    xccl_cuda_mem_component_request_t *reqs;

    xccl_cuda_mem_component.stream = 0;
    reqs = (xccl_cuda_mem_component_request_t*)malloc(NUM_REQUESTS*sizeof(*reqs));
    if (reqs == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    xccl_cuda_mem_component.requests = reqs;

    return XCCL_OK;
}

static xccl_status_t xccl_cuda_init_resources() {
    int i;

    CUDACHECK(cudaStreamCreateWithFlags(&xccl_cuda_mem_component.stream,
                                        cudaStreamNonBlocking));
    for (i = 0; i < NUM_REQUESTS; i++) {
        xccl_cuda_mem_component.requests[i].is_free = 1;
        xccl_cuda_mem_component.requests[i].super.component_id = UCS_MEMORY_TYPE_CUDA;
        // CUDACHECK(cudaEventCreateWithFlags(&xccl_cuda_mem_component.requests[i].event,
        //             cudaEventBlockingSync | cudaEventDisableTiming));
        CUDACHECK(cudaEventCreate(&xccl_cuda_mem_component.requests[i].event));
        CUDACHECK(cudaEventCreate(&xccl_cuda_mem_component.requests[i].start));
    }

    return XCCL_OK;
}

static xccl_status_t xccl_cuda_mem_alloc(void **ptr, size_t len)
{
    CUDACHECK(cudaMalloc(ptr, len));
    return XCCL_OK;
}

static xccl_status_t xccl_cuda_mem_free(void *ptr)
{
    CUDACHECK(cudaFree(ptr));
    return XCCL_OK;
}

xccl_status_t xccl_cuda_reduce_impl(void *sbuf1, void *sbuf2, void *target,
                                    size_t count, xccl_dt_t dtype, xccl_op_t op,
                                    cudaStream_t stream);

xccl_status_t xccl_cuda_reduce(void *sbuf1, void *sbuf2, void *target,
                               size_t count, xccl_dt_t dtype, xccl_op_t op)
{
    if (xccl_cuda_mem_component.stream == 0) {
        if(xccl_cuda_init_resources() != XCCL_OK) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    return xccl_cuda_reduce_impl(sbuf1, sbuf2, target, count, dtype, op,
                                 xccl_cuda_mem_component.stream);
}

xccl_status_t xccl_cuda_reduce_multi_impl(void *sbuf1, void *sbuf2, void *rbuf,
                                         size_t count, size_t size, size_t stride,
                                         xccl_dt_t dtype, xccl_op_t op,
                                         cudaStream_t stream);

xccl_status_t xccl_cuda_reduce_multi(void *sbuf1, void *sbuf2, void *rbuf,
                                     size_t count, size_t size, size_t stride,
                                     xccl_dt_t dtype, xccl_op_t op)
{
    if (xccl_cuda_mem_component.stream == 0) {
        if(xccl_cuda_init_resources() != XCCL_OK) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    return xccl_cuda_reduce_multi_impl(sbuf1, sbuf2, rbuf, count, size, stride,
                                       dtype, op,
                                       xccl_cuda_mem_component.stream);
}

xccl_status_t xccl_cuda_mem_type(void *ptr, ucs_memory_type_t *mem_type) {
    struct      cudaPointerAttributes attr;
    cudaError_t err;

    err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return XCCL_ERR_UNSUPPORTED;
    }

#if CUDART_VERSION >= 10000
    if (attr.type == cudaMemoryTypeDevice) {
#else
    if (attr.memoryType == cudaMemoryTypeDevice) {
#endif
        *mem_type = UCS_MEMORY_TYPE_CUDA;
    }
    else {
        *mem_type = UCS_MEMORY_TYPE_HOST;
    }

    return XCCL_OK;
}

static xccl_status_t
xccl_cuda_get_free_request(xccl_cuda_mem_component_request_t **request) {
    int i;
    xccl_cuda_mem_component_request_t *req;
    for (i = 0; i < NUM_REQUESTS; i++) {
        req = xccl_cuda_mem_component.requests + i;
        if (req->is_free) {
            req->is_free = 0;
            *request = req;
            return XCCL_OK;
        }
    }

    return XCCL_ERR_NO_RESOURCE;
}

static xccl_status_t xccl_cuda_copy_async(void *dst, ucs_memory_type_t dst_mtype,
                                          void *src, ucs_memory_type_t src_mtype,
                                          size_t length,
                                          xccl_mem_component_request_t **req)
{
    enum cudaMemcpyKind kind;
    cudaError_t err;
    xccl_cuda_mem_component_request_t *request;

    assert(((dst_mtype == UCS_MEMORY_TYPE_CUDA) || (dst_mtype == UCS_MEMORY_TYPE_HOST)));
    assert(((src_mtype == UCS_MEMORY_TYPE_CUDA) || (src_mtype == UCS_MEMORY_TYPE_HOST)));

    if (src_mtype == UCS_MEMORY_TYPE_CUDA) {
        if (dst_mtype == UCS_MEMORY_TYPE_CUDA) {
            kind = cudaMemcpyDeviceToDevice;
        } else {
            kind = cudaMemcpyDeviceToHost;
        }
    } else {
        if (dst_mtype == UCS_MEMORY_TYPE_CUDA) {
            kind = cudaMemcpyHostToDevice;
        } else {
            kind = cudaMemcpyHostToHost;
        }
    }

    if (xccl_cuda_mem_component.stream == 0) {
        if(xccl_cuda_init_resources() != XCCL_OK) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    if (xccl_cuda_get_free_request(&request) != XCCL_OK) {
        fprintf(stderr, "cuda memcpy failed: no free requests\n");
        return XCCL_ERR_NO_RESOURCE;
    }
    request->super.status = XCCL_INPROGRESS;
    CUDACHECK(cudaEventRecord(request->start, xccl_cuda_mem_component.stream));
    CUDACHECK(cudaMemcpyAsync(dst, src, length, kind, xccl_cuda_mem_component.stream));
    CUDACHECK(cudaEventRecord(request->event, xccl_cuda_mem_component.stream));

    *req = &request->super;
    return XCCL_OK;
}

static float total_time = 0;
static int n_events = 0;

static xccl_status_t xccl_cuda_test_req(xccl_mem_component_request_t *req)
{
    cudaError_t err;
    xccl_cuda_mem_component_request_t *request;
    float ms;

    request = ucs_derived_of(req, xccl_cuda_mem_component_request_t);
    if (request->super.status == XCCL_INPROGRESS) {
        err = cudaEventQuery(request->event);
        switch(err) {
        case cudaSuccess:
            cudaEventElapsedTime(&ms, request->start, request->event);
            total_time += ms;
            n_events++;
            request->super.status = XCCL_OK;
            break;
        case cudaErrorNotReady:
            break;
        default:
            fprintf(stderr, "cuda test failed: %d %s\n", err, cudaGetErrorName(err));
            request->super.status = XCCL_ERR_NO_MESSAGE;
        }
    }

    return request->super.status;
}

static xccl_status_t xccl_cuda_free_req(xccl_mem_component_request_t *req)
{
    cudaError_t err;
    xccl_cuda_mem_component_request_t *request;

    request = ucs_derived_of(req, xccl_cuda_mem_component_request_t);
    if (cudaEventQuery(request->event) != cudaSuccess) {
        fprintf(stderr, "calling free req before operation is done\n");
        return XCCL_ERR_NO_MESSAGE;
    }

    request->is_free = 1;

    return XCCL_OK;
}


static void xccl_cuda_close()
{
    int i;

    if (xccl_cuda_mem_component.stream != 0) {
        fprintf(stderr, "avg time: %f\n", total_time/n_events);
        cudaStreamDestroy(xccl_cuda_mem_component.stream);
        for (i = 0; i < NUM_REQUESTS; i++) {
            cudaEventDestroy(xccl_cuda_mem_component.requests[i].event);
            cudaEventDestroy(xccl_cuda_mem_component.requests[i].start);
        }

    }
}

xccl_cuda_mem_component_t xccl_cuda_mem_component = {
    xccl_cuda_open,
    xccl_cuda_mem_alloc,
    xccl_cuda_mem_free,
    xccl_cuda_mem_type,
    xccl_cuda_copy_async,
    xccl_cuda_reduce,
    xccl_cuda_reduce_multi,
    xccl_cuda_test_req,
    xccl_cuda_free_req,
    xccl_cuda_close
};
