/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_MEM_COMPONENT_H_
#define XCCL_MEM_COMPONENT_H_

#include "api/xccl.h"
#include <ucs/memory/memory_type.h>

typedef struct xccl_mem_component_buf {
    void   *buf;
    size_t size;
    int    used;
} xccl_mem_component_buf_t;

typedef struct xccl_mem_component_request {
    ucs_memory_type_t component_id;
    uint64_t          tag;
    xccl_status_t     status;
} xccl_mem_component_request_t;

typedef struct xccl_mem_component {
    xccl_status_t (*open)();
    xccl_status_t (*mem_alloc)(void **ptr, size_t len);
    xccl_status_t (*mem_free)(void *ptr);
    xccl_status_t (*mem_type)(void *ptr, ucs_memory_type_t *mem_type);
    xccl_status_t (*copy_async)(void *dst, ucs_memory_type_t dst_mtype,
                                void *src, ucs_memory_type_t src_mtype,
                                size_t length,
                                xccl_mem_component_request_t **req);
    xccl_status_t (*reduce)(void *sbuf1, void *sbuf2, void *target,
                            size_t count, xccl_dt_t dtype, xccl_op_t op);
    xccl_status_t (*reduce_multi)(void *sbuf1, void *sbuf2, void *rbuf,
                                  size_t count, size_t size, size_t stride,
                                  xccl_dt_t dtype, xccl_op_t op);
    xccl_status_t (*test_req)(xccl_mem_component_request_t *req);
    xccl_status_t (*free_req)(xccl_mem_component_request_t *req);
    void          (*close)();
    void                     *dlhandle;
    xccl_mem_component_buf_t cache;
} xccl_mem_component_t;


xccl_status_t xccl_mem_component_init(const char* components_path);

xccl_status_t xccl_mem_component_alloc(void **ptr, size_t len,
                                       ucs_memory_type_t mem_type);

xccl_status_t xccl_mem_component_free(void *ptr, ucs_memory_type_t mem_type);

xccl_status_t xccl_mem_component_type(void *ptr, ucs_memory_type_t *mem_type);

xccl_status_t xccl_mem_component_reduce(void *sbuf1, void *sbuf2, void *target,
                                        size_t count, xccl_dt_t dtype,
                                        xccl_op_t op, ucs_memory_type_t mem_type);

xccl_status_t xccl_mem_component_copy_async(void *dst, ucs_memory_type_t dst_mtype,
                                            void *src, ucs_memory_type_t src_mtype,
                                            size_t length,
                                            xccl_mem_component_request_t **req);

xccl_status_t xccl_mem_component_test_request(xccl_mem_component_request_t *req);

xccl_status_t xccl_mem_component_free_request(xccl_mem_component_request_t *req);

/*
 * Performs reduction of multiple vectors and stores result to rbuf
 * rbuf = sbuf1 + sbuf2{0} + sbuf2{1} + sbuf2{count-1}
 * count  - number of vectors in sbuf2
 * size   - size of each verctor
 * stride - offset between vectors in sbuf2
 */

xccl_status_t
xccl_mem_component_reduce_multi(void *sbuf1, void *sbuf2, void *rbuf, size_t count,
                                size_t size, size_t stride, xccl_dt_t dtype,
                                xccl_op_t op, ucs_memory_type_t mem_type);

void xccl_mem_component_free_cache();

void xccl_mem_component_finalize();

#endif
