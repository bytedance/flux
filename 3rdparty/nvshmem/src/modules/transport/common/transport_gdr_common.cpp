/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "transport_gdr_common.h"
#include <dlfcn.h>
#include <string.h>

#include "transport_common.h"

int nvshmemi_gdrapi_compile_time_major_version;
int nvshmemi_gdrapi_compile_time_minor_version;

bool nvshmemt_gdrcopy_ftable_init(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle, int log_level) {
    bool use_gdrcopy = true;
    void *local_gdrcopy_handle;
    int major, minor;
    nvshmemi_gdrapi_compile_time_major_version = GDR_API_MAJOR_VERSION;
    nvshmemi_gdrapi_compile_time_minor_version = GDR_API_MINOR_VERSION;

    *gdrcopy_handle = dlopen("libgdrapi.so.2", RTLD_LAZY);
    if (!*gdrcopy_handle) {
        INFO(log_level, "GDRCopy library not found. disabling GDRCopy.\n");
        use_gdrcopy = false;
        goto out;
    } else {
        local_gdrcopy_handle = *gdrcopy_handle;
        LOAD_SYM(local_gdrcopy_handle, "gdr_runtime_get_version",
                 gdrcopy_ftable->runtime_get_version);
        if (!gdrcopy_ftable->runtime_get_version) {
            INFO(log_level, "GDRCopy library is older than v2.0. Disabling GDRCopy.");
            use_gdrcopy = false;
            goto out;
        }
        int gdrapi_runtime_major_version, gdrapi_runtime_minor_version;
        gdrcopy_ftable->runtime_get_version(&gdrapi_runtime_major_version,
                                            &gdrapi_runtime_minor_version);
        if (gdrapi_runtime_major_version != nvshmemi_gdrapi_compile_time_major_version ||
            gdrapi_runtime_minor_version < nvshmemi_gdrapi_compile_time_minor_version) {
            INFO(log_level,
                 "GDRCopy library version is not compatible with gdrapi.h (%d.%d) used during "
                 "compilation. "
                 "Disabling GDRCopy.\n",
                 nvshmemi_gdrapi_compile_time_major_version,
                 nvshmemi_gdrapi_compile_time_minor_version);
            use_gdrcopy = false;
            goto out;
        }
        LOAD_SYM(local_gdrcopy_handle, "gdr_driver_get_version",
                 gdrcopy_ftable->driver_get_version);
        LOAD_SYM(local_gdrcopy_handle, "gdr_open", gdrcopy_ftable->open);
        LOAD_SYM(local_gdrcopy_handle, "gdr_close", gdrcopy_ftable->close);
        LOAD_SYM(local_gdrcopy_handle, "gdr_pin_buffer", gdrcopy_ftable->pin_buffer);
        LOAD_SYM(local_gdrcopy_handle, "gdr_unpin_buffer", gdrcopy_ftable->unpin_buffer);
        LOAD_SYM(local_gdrcopy_handle, "gdr_map", gdrcopy_ftable->map);
        LOAD_SYM(local_gdrcopy_handle, "gdr_unmap", gdrcopy_ftable->unmap);
        LOAD_SYM(local_gdrcopy_handle, "gdr_get_info", gdrcopy_ftable->get_info);
        LOAD_SYM(local_gdrcopy_handle, "gdr_copy_from_mapping", gdrcopy_ftable->copy_from_mapping);
        LOAD_SYM(local_gdrcopy_handle, "gdr_copy_to_mapping", gdrcopy_ftable->copy_to_mapping);
    }

    *gdr_desc = gdrcopy_ftable->open();
    if (!*gdr_desc) {
        INFO(log_level, "GDRCopy open call failed, disabling GDRCopy.\n");
        use_gdrcopy = false;
        goto out;
    }

    gdrcopy_ftable->driver_get_version(*gdr_desc, &major, &minor);
    INFO(log_level, "GDR driver version: (%d, %d)", major, minor);

out:
    if (!use_gdrcopy) {
        if (*gdrcopy_handle) {
            dlclose(*gdrcopy_handle);
            *gdrcopy_handle = NULL;
        }

        memset(gdrcopy_ftable, 0, sizeof(struct gdrcopy_function_table));
    }

    return use_gdrcopy;
}

void nvshmemt_gdrcopy_ftable_fini(struct gdrcopy_function_table *gdrcopy_ftable, gdr_t *gdr_desc,
                                  void **gdrcopy_handle) {
    if (gdrcopy_ftable->close && gdr_desc) {
        gdrcopy_ftable->close(*gdr_desc);
    }

    if (gdrcopy_handle && *gdrcopy_handle) {
        dlclose(*gdrcopy_handle);
        *gdrcopy_handle = NULL;
    }
}
