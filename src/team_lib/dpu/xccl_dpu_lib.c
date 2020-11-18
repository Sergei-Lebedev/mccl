/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "xccl_dpu_lib.h"
#include <ucs/memory/memory_type.h>

static ucs_config_field_t xccl_team_lib_dpu_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_team_lib_dpu_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t xccl_tl_dpu_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_dpu_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {NULL}
};

static xccl_status_t xccl_dpu_lib_open(xccl_team_lib_h self,
                                       xccl_team_lib_config_t *config)
{
    xccl_team_lib_dpu_t        *tl  = ucs_derived_of(self, xccl_team_lib_dpu_t);
    xccl_team_lib_dpu_config_t *cfg = ucs_derived_of(config, xccl_team_lib_dpu_config_t);

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_DPU");
    xccl_dpu_debug("Team DPU opened");
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }

    return XCCL_OK;
}


static xccl_status_t
xccl_dpu_context_create(xccl_team_lib_h lib, xccl_context_params_t *params,
                        xccl_tl_context_config_t *config,
                        xccl_tl_context_t **context)
{
    xccl_dpu_context_t *ctx = malloc(sizeof(*ctx));

    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    *context = &ctx->super;

    return XCCL_OK;
}

static xccl_status_t
xccl_dpu_context_destroy(xccl_tl_context_t *context)
{
    xccl_dpu_context_t *team_dpu_ctx =
        ucs_derived_of(context, xccl_dpu_context_t);

    free(team_dpu_ctx);

    return XCCL_OK;
}

static xccl_status_t
xccl_dpu_team_create_post(xccl_tl_context_t *context,
                          xccl_team_params_t *params,
                          xccl_tl_team_t **team)
{
    xccl_dpu_team_t *dpu_team = malloc(sizeof(*dpu_team));
    XCCL_TEAM_SUPER_INIT(dpu_team->super, context, params);

    *team = &dpu_team->super;
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_team_create_test(xccl_tl_team_t *team)
{
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_team_destroy(xccl_tl_team_t *team)
{
    xccl_dpu_team_t *dpu_team = ucs_derived_of(team, xccl_dpu_team_t);

    free(team);
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_init(xccl_coll_op_args_t *coll_args,
                                              xccl_tl_coll_req_t **request,
                                              xccl_tl_team_t *team)
{
    xccl_dpu_info("Collective init");
    *request = (xccl_tl_coll_req_t*)malloc(sizeof(xccl_tl_coll_req_t));
    if (*request == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    (*request)->lib = &xccl_team_lib_dpu.super;

    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_dpu_info("Collective post");
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_dpu_info("Collective wait");
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_dpu_info("Collective test");
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_dpu_info("Collective finalize");
    return XCCL_OK;
}

xccl_team_lib_dpu_t xccl_team_lib_dpu = {
    .super.name                   = "dpu",
    .super.id                     = XCCL_TL_DPU,
    .super.priority               = 90,
    .super.team_lib_config        =
    {
        .name                     = "DPU team library",
        .prefix                   = "TEAM_DPU_",
        .table                    = xccl_team_lib_dpu_config_table,
        .size                     = sizeof(xccl_team_lib_dpu_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "DPU tl context",
        .prefix                  = "TEAM_DPU_",
        .table                   = xccl_tl_dpu_context_config_table,
        .size                    = sizeof(xccl_tl_dpu_context_config_t),
    },
    .super.params.reproducible    = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode     = XCCL_THREAD_MODE_SINGLE |
                                    XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage      = XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    .super.params.coll_types      = XCCL_COLL_CAP_ALLREDUCE,
    .super.mem_types              = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode        = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create    = xccl_dpu_context_create,
    .super.team_context_destroy   = xccl_dpu_context_destroy,
    .super.team_context_progress  = NULL,
    .super.team_create_post       = xccl_dpu_team_create_post,
    .super.team_create_test       = xccl_dpu_team_create_test,
    .super.team_destroy           = xccl_dpu_team_destroy,
    .super.team_lib_open          = xccl_dpu_lib_open,
    .super.collective_init        = xccl_dpu_collective_init,
    .super.collective_post        = xccl_dpu_collective_post,
    .super.collective_wait        = xccl_dpu_collective_wait,
    .super.collective_test        = xccl_dpu_collective_test,
    .super.collective_finalize    = xccl_dpu_collective_finalize,
    .super.global_mem_map_start   = NULL,
    .super.global_mem_map_test    = NULL,
    .super.global_mem_unmap       = NULL,
};
