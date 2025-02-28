/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"
#include "internal/bootstrap_host/nvshmemi_bootstrap.h"
#include "non_abi/nvshmemx_error.h"

#define GET_SYMBOL(lib_handle, name, var, status)                                                \
    do {                                                                                         \
        void **var_ptr = (void **)&(var);                                                        \
        void *tmp = (void *)dlsym(lib_handle, name);                                             \
        NVSHMEMI_NULL_ERROR_JMP(tmp, status, NVSHMEMX_ERROR_INTERNAL, out,                       \
                                "Bootstrap failed to get symbol '%s'\n\t%s\n", name, dlerror()); \
        *var_ptr = tmp;                                                                          \
    } while (0)

static void *plugin_hdl = nullptr;
static char *plugin_name = nullptr;

void _bootstrap_loader_fini_helper(void *plugin_hdl, char *plugin_name) {
    if (plugin_hdl != nullptr) {
        dlclose(plugin_hdl);
        plugin_hdl = nullptr;
    }

    if (plugin_name != nullptr) {
        free(plugin_name);
        plugin_name = nullptr;
    }
}

int bootstrap_loader_finalize(bootstrap_handle_t *handle) {
    int status = handle->finalize(handle);

    if (status != 0)
        NVSHMEMI_ERROR_PRINT("Bootstrap plugin finalize failed for '%s'\n", plugin_name);

    dlclose(plugin_hdl);
    plugin_hdl = nullptr;
    free(plugin_name);
    plugin_name = nullptr;

    return 0;
}

static int _bootstrap_loader_init_helper(const char *plugin, bootstrap_handle_t *handle) {
    int status = 0;

    dlerror(); /* Clear any existing error */
    if (plugin_name == nullptr) {
        plugin_name = strdup(plugin);
    }

    if (plugin_hdl == nullptr) {
        plugin_hdl = dlopen(plugin, RTLD_NOW);
    }

    NVSHMEMI_NULL_ERROR_JMP(plugin_hdl, status, -1, error, "Bootstrap unable to load '%s'\n\t%s\n",
                            plugin, dlerror());

    dlerror(); /* Clear any existing error */
    goto out;

error:
    _bootstrap_loader_fini_helper(plugin_hdl, plugin_name);
out:
    return status;
}

int bootstrap_loader_preinit(const char *plugin, bootstrap_handle_t *handle) {
    int status = 0;
    int (*bootstrap_plugin_preinitops)(bootstrap_handle_t * handle, int nvshmem_version);
    status = _bootstrap_loader_init_helper(plugin, handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap library dlopen failed for %s\n", plugin);
    GET_SYMBOL(plugin_hdl, "nvshmemi_bootstrap_plugin_pre_init", bootstrap_plugin_preinitops,
               status);
    status = bootstrap_plugin_preinitops(handle, NVSHMEMI_BOOTSTRAP_ABI_VERSION);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap plugin preinit failed for '%s'\n", plugin);
    goto out;
error:
    _bootstrap_loader_fini_helper(plugin_hdl, plugin_name);
out:
    return (status);
}

int bootstrap_loader_init(const char *plugin, void *arg, bootstrap_handle_t *handle) {
    int status = 0;
    int (*bootstrap_plugin_initops)(void *arg, bootstrap_handle_t *handle, int nvshmem_version);
    status = _bootstrap_loader_init_helper(plugin, handle);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap library dlopen failed for %s\n", plugin);
    GET_SYMBOL(plugin_hdl, "nvshmemi_bootstrap_plugin_init", bootstrap_plugin_initops, status);
    status = bootstrap_plugin_initops(arg, handle, NVSHMEMI_BOOTSTRAP_ABI_VERSION);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                          "Bootstrap plugin init failed for '%s'\n", plugin);
    if (NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(NVSHMEMI_BOOTSTRAP_ABI_VERSION) <
        NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(handle->version)) {
        NVSHMEMI_ERROR_PRINT(
            "The selected bootstrap returned an incompatible version %d. "
            "Expected version %d\n",
            NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(handle->version),
            NVSHMEM_BOOTSTRAP_MAJOR_MINOR_VERSION(NVSHMEMI_BOOTSTRAP_ABI_VERSION));
        goto error;
    }
    goto out;
error:
    _bootstrap_loader_fini_helper(plugin_hdl, plugin_name);
out:
    return (status);
}
