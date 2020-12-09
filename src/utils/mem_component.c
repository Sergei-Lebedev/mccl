#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "mem_component.h"
#include "reduce.h"
#include "utils/xccl_log.h"

static xccl_mem_component_t *mem_components[UCS_MEMORY_TYPE_LAST];
extern xccl_config_t xccl_lib_global_config;

xccl_status_t xccl_mem_component_init(const char* components_path)
{
    void          *handle;
    char          *mem_comp_path;
    size_t        mem_comp_path_len;
    int           mt;
    xccl_status_t status;

    mem_comp_path_len = strlen(components_path) + 32;
    mem_comp_path     = (char*)malloc(mem_comp_path_len);
    if (mem_comp_path == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    for (mt = UCS_MEMORY_TYPE_HOST + 1; mt < UCS_MEMORY_TYPE_LAST; mt++) {
        snprintf(mem_comp_path, mem_comp_path_len, "%s/xccl_%s_mem_component.so",
                 components_path, ucs_memory_type_names[mt]);
        handle = dlopen(mem_comp_path, RTLD_LAZY);
        if (!handle) {
            continue;
        }
        snprintf(mem_comp_path, mem_comp_path_len, "xccl_%s_mem_component",
                 ucs_memory_type_names[mt]);

        mem_components[mt] = (xccl_mem_component_t*)dlsym(handle, mem_comp_path);
        mem_components[mt]->dlhandle   = handle;
        mem_components[mt]->cache.size = xccl_lib_global_config.mem_component_cache_size;
        mem_components[mt]->cache.used = 0;
        mem_components[mt]->cache.buf  = NULL;
        status = mem_components[mt]->open();
        if (status != XCCL_OK) {
            xccl_warn("failed to open %s mem component", ucs_memory_type_names[mt]);
            dlclose(mem_components[mt]->dlhandle);
            mem_components[mt] = NULL;
            continue;
        }
        xccl_debug("%s mem component found", ucs_memory_type_names[mt]);
    }

    free(mem_comp_path);
    return XCCL_OK;
}

xccl_status_t xccl_mem_component_alloc(void **ptr, size_t len,
                                       ucs_memory_type_t mt)
{
    xccl_status_t st;

    if (mt == UCS_MEMORY_TYPE_HOST) {
        *ptr = malloc(len);
        if (!(*ptr)) {
            return XCCL_ERR_NO_MEMORY;
        }
        return XCCL_OK;
    }

    if (mem_components[mt] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    if ((mem_components[mt]->cache.used == 0) &&
        (mem_components[mt]->cache.size >= len)) {

        if (mem_components[mt]->cache.buf == NULL) {
            st = mem_components[mt]->mem_alloc(&mem_components[mt]->cache.buf,
                                               mem_components[mt]->cache.size);
            if (st != XCCL_OK) {
                return st;
            }
        }

        *ptr = mem_components[mt]->cache.buf;
        mem_components[mt]->cache.used = 1;
        return XCCL_OK;
    }

    return mem_components[mt]->mem_alloc(ptr, len);
}

xccl_status_t xccl_mem_component_free(void *ptr,
                                      ucs_memory_type_t mem_type)
{
    if (mem_type == UCS_MEMORY_TYPE_HOST) {
        free(ptr);
        return XCCL_OK;
    }

    if (mem_components[mem_type] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    if (ptr == mem_components[mem_type]->cache.buf) {
        mem_components[mem_type]->cache.used = 0;
        return XCCL_OK;
    }

    return mem_components[mem_type]->mem_free(ptr);
}

xccl_status_t xccl_mem_component_reduce(void *sbuf1, void *sbuf2, void *target,
                                        size_t count, xccl_dt_t dtype,
                                        xccl_op_t op, ucs_memory_type_t mem_type)
{
    if (mem_type == UCS_MEMORY_TYPE_HOST) {
        return xccl_dt_reduce(sbuf1, sbuf2, target, count, dtype, op);
    }

    if (mem_components[mem_type] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    return mem_components[mem_type]->reduce(sbuf1, sbuf2, target, count, dtype, op);
}

xccl_status_t
xccl_mem_component_reduce_multi(void *sbuf1, void *sbuf2, void *rbuf, size_t count,
                                size_t size, size_t stride, xccl_dt_t dtype,
                                xccl_op_t op, ucs_memory_type_t mem_type)
{
    int i;

    if (size == 0) {
        return XCCL_OK;
    }

    if (mem_type == UCS_MEMORY_TYPE_HOST) {
        xccl_dt_reduce(sbuf1, sbuf2, rbuf, size, dtype, op);
        for (i = 1; i < count; i++) {
            xccl_dt_reduce((void*)((ptrdiff_t)sbuf2 + stride*i), rbuf,
                           rbuf, size, dtype, op);
        }
        return XCCL_OK;
    }

    if (mem_components[mem_type] == NULL) {
         return XCCL_ERR_UNSUPPORTED;
    }

    return mem_components[mem_type]->reduce_multi(sbuf1, sbuf2, rbuf, count,
                                                  size, stride, dtype, op);
}

xccl_status_t xccl_mem_component_copy_async(void *dst, ucs_memory_type_t dst_mtype,
                                            void *src, ucs_memory_type_t src_mtype,
                                            size_t length,
                                            xccl_mem_component_request_t **req)
{
    if ((dst_mtype == UCS_MEMORY_TYPE_HOST) && (src_mtype == UCS_MEMORY_TYPE_HOST))
    {
        memcpy(dst, src, length);
        *req = (xccl_mem_component_request_t*)malloc(sizeof(*req));
        (*req)->status = XCCL_OK;
        (*req)->component_id = UCS_MEMORY_TYPE_HOST;
        return XCCL_OK;
    }

    if (((dst_mtype == UCS_MEMORY_TYPE_CUDA) || (src_mtype == UCS_MEMORY_TYPE_CUDA)) &&
        (mem_components[UCS_MEMORY_TYPE_CUDA] != NULL))
    {
        return mem_components[UCS_MEMORY_TYPE_CUDA]->copy_async(
            dst, dst_mtype, src, src_mtype, length, req);
    }

    return XCCL_ERR_UNSUPPORTED;
}

xccl_status_t xccl_mem_component_test_request(xccl_mem_component_request_t *req)
{
    if (req == NULL) {
        return XCCL_OK;
    }

    if (req->component_id == UCS_MEMORY_TYPE_HOST) {
        return XCCL_OK;
    }

    return mem_components[req->component_id]->test_req(req);
}

xccl_status_t xccl_mem_component_free_request(xccl_mem_component_request_t *req)
{
    if (req == NULL) {
        return XCCL_OK;
    }
    if (req->component_id == UCS_MEMORY_TYPE_HOST) {
        free(req);
        return XCCL_OK;
    }
    return mem_components[req->component_id]->free_req(req);
}


xccl_status_t xccl_mem_component_type(void *ptr, ucs_memory_type_t *mem_type)
{
    xccl_status_t st;
    int           mt;

    *mem_type = UCS_MEMORY_TYPE_LAST;

    for(mt = UCS_MEMORY_TYPE_HOST+1; mt < UCS_MEMORY_TYPE_LAST; mt++) {
        if (mem_components[mt] != NULL) {
            st = mem_components[mt]->mem_type(ptr, mem_type);
            if ((st != XCCL_OK) || (*mem_type == UCS_MEMORY_TYPE_LAST)) {
                /* this mem_component wasn't able to detect memory type
                 * continue to next
                 */
                continue;
            }
            return st;
        }
    }

    xccl_debug("Assuming mem_type host");
    *mem_type = UCS_MEMORY_TYPE_HOST;

    return XCCL_OK;
}

void xccl_mem_component_free_cache()
{
    int mt;

    for(mt = UCS_MEMORY_TYPE_HOST + 1; mt < UCS_MEMORY_TYPE_LAST; mt++) {
        if (mem_components[mt] != NULL) {
            if ((mem_components[mt]->cache.buf) &&
                (!mem_components[mt]->cache.used)) {
                mem_components[mt]->mem_free(mem_components[mt]->cache.buf);
            }
        }
    }
}

void xccl_mem_component_finalize()
{
    int mt;

    for(mt = UCS_MEMORY_TYPE_HOST + 1; mt < UCS_MEMORY_TYPE_LAST; mt++) {
        if (mem_components[mt] != NULL) {
            mem_components[mt]->close();
            dlclose(mem_components[mt]->dlhandle);
        }
    }
}
