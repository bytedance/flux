#ifndef NVML_WRAP_H
#define NVML_WRAP_H

#include <cuda.h>
#include <nvml.h>

/* Copied from CUDA 12.4 NVML header. */
#if ((NVML_API_VERSION < 12) || (CUDA_VERSION < 12040))

#ifndef NVML_GPU_FABRIC_STATE_COMPLETED
#define NVML_GPU_FABRIC_STATE_COMPLETED 3
#endif

#ifndef nvmlGpuFabricInfo_v2
#define nvmlGpuFabricInfo_v2 (unsigned int)(sizeof(nvmlGpuFabricInfo_v2_t) | (2 << 24U))
#endif

#ifndef NVML_GPU_FABRIC_UUID_LEN
#define NVML_GPU_FABRIC_UUID_LEN 16
#endif

typedef unsigned char nvmlGpuFabricState_t;
typedef struct {
    unsigned int version;  //!< Structure version identifier (set to \ref nvmlGpuFabricInfo_v2)
    unsigned char
        clusterUuid[NVML_GPU_FABRIC_UUID_LEN];  //!< Uuid of the cluster to which this GPU belongs
    nvmlReturn_t
        status;  //!< Error status, if any. Must be checked only if state returns "complete".
    unsigned int cliqueId;       //!< ID of the fabric clique to which this GPU belongs
    nvmlGpuFabricState_t state;  //!< Current state of GPU registration process
    unsigned int healthMask;     //!< GPU Fabric health Status Mask
} nvmlGpuFabricInfo_v2_t;

typedef nvmlGpuFabricInfo_v2_t nvmlGpuFabricInfoV_t;
#endif
/* end NVML Header defs. */

struct nvml_function_table {
    nvmlReturn_t (*nvmlInit)(void);
    nvmlReturn_t (*nvmlShutdown)(void);
    nvmlReturn_t (*nvmlDeviceGetHandleByPciBusId)(const char *pciBusId, nvmlDevice_t *device);
    nvmlReturn_t (*nvmlDeviceGetP2PStatus)(nvmlDevice_t device1, nvmlDevice_t device2,
                                           nvmlGpuP2PCapsIndex_enum caps,
                                           nvmlGpuP2PStatus_t *p2pStatus);
    nvmlReturn_t (*nvmlDeviceGetGpuFabricInfoV)(nvmlDevice_t device, nvmlGpuFabricInfoV_t *info);
};

int nvshmemi_nvml_ftable_init(struct nvml_function_table *nvml_ftable, void **nvml_handle);
void nvshmemi_nvml_ftable_fini(struct nvml_function_table *nvml_ftable, void **nvml_handle);

#endif