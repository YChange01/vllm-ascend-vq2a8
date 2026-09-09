// User-supplied runtime declarations, not the installed CANN SDK. See README.md.
#ifndef __HACL_RUNTIME_RT_H__
#define __HACL_RUNTIME_RT_H__
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __declspec(dllexport)
#else
#define RTS_API
#endif
#endif
typedef void *rtStream_t;
typedef enum tagRtError {
    RT_ERROR_NONE = 0x0, RT_ERROR_INVALID_VALUE = 0x1, RT_ERROR_MEMORY_ALLOCATION = 0x2,
    RT_ERROR_INVALID_RESOURCE_HANDLE = 0x3, RT_ERROR_INVALID_DEVICE_POINTER = 0x4,
    RT_ERROR_INVALID_MEMCPY_DIRECTION = 0x5, RT_ERROR_INVALID_DEVICE = 0x6, RT_ERROR_NO_DEVICE = 0x7,
    RT_ERROR_CMD_OCCUPY_FAILURE = 0x8, RT_ERROR_SET_SIGNAL_FAILURE = 0x9, RT_ERROR_UNSET_SIGNAL_FAILURE = 0xA,
    RT_ERROR_OPEN_FILE_FAILURE = 0xB, RT_ERROR_WRITE_FILE_FAILURE = 0xC, RT_ERROR_MEMORY_ADDRESS_UNALIGNED = 0xD,
    RT_ERROR_DRV_ERR = 0xE, RT_ERROR_LOST_HEARTBEAT = 0xF, RT_ERROR_REPORT_TIMEOUT = 0x10,
    RT_ERROR_NOT_READY = 0x11, RT_ERROR_DATA_OPERATION_FAIL = 0x12, RT_ERROR_INVALID_L2_INSTR_SIZE = 0x13,
    RT_ERROR_DEVICE_PROC_HANG_OUT = 0x14, RT_ERROR_DEVICE_POWER_UP_FAIL = 0x15,
    RT_ERROR_DEVICE_POWER_DOWN_FAIL = 0x16, RT_ERROR_FEATURE_NOT_SUPPROT = 0x17,
    RT_ERROR_KERNEL_DUPLICATE = 0x18, RT_ERROR_MODEL_STREAM_EXE_FAILED = 0x91,
    RT_ERROR_MODEL_LOAD_FAILED = 0x94, RT_ERROR_END_OF_SEQUENCE = 0x95, RT_ERROR_NO_STREAM_CB_REG = 0x96,
    RT_ERROR_DATA_DUMP_LOAD_FAILED = 0x97,
    RT_ERROR_CALLBACK_THREAD_UNSUBSTRIBE = 0x98, RT_ERROR_RESERVED
} rtError_t;
typedef struct tagRtDevBinary { uint32_t magic; uint32_t version; const void *data; uint64_t length; } rtDevBinary_t;
typedef struct tagRtSmData {
    uint64_t L2_mirror_addr;
    uint32_t L2_data_section_size;
    uint8_t L2_preload;
    uint8_t modified;
    uint8_t priority;
    int8_t prev_L2_page_offset_base;
    uint8_t L2_page_offset_base;
    uint8_t L2_load_to_ddr;
    uint8_t reserved[2];
} rtSmData_t;
typedef struct tagRtSmCtrl {
    rtSmData_t data[8]; uint64_t size; uint8_t remap[64]; uint8_t l2_in_main; uint8_t reserved[3];
} rtSmDesc_t;
#define RT_DEV_BINARY_MAGIC_PLAIN 0xabceed50
#define RT_DEV_BINARY_MAGIC_PLAIN_AICPU 0xabceed51
#define RT_DEV_BINARY_MAGIC_PLAIN_AIVEC 0xabceed52
#define RT_DEV_BINARY_MAGIC_ELF 0x43554245
#define RT_DEV_BINARY_MAGIC_ELF_AICPU 0x41415243
#define RT_DEV_BINARY_MAGIC_ELF_AIVEC 0x41415246
RTS_API rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle);
RTS_API rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName,
                                   const void *devFunc, uint32_t funcMode);
RTS_API rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                               rtSmDesc_t *smDesc, rtStream_t stream);
typedef struct tagRtArgsEx {
    void *args; void *hostInputInfoPtr; uint32_t argsSize; uint16_t tilingAddrOffset;
    uint16_t tilingDataOffset; uint16_t hostInputInfoNum; uint8_t hasTiling; uint8_t isNoNeedH2DCopy; uint8_t reserved[4];
} rtArgsEx_t;
RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                       rtSmDesc_t *smDesc, rtStream_t stream, uint32_t flag = 0);
#define RT_MEMORY_DEFAULT ((uint32_t)0x0)
#define RT_MEMORY_HBM ((uint32_t)0x2)
#define RT_MEMORY_DDR ((uint32_t)0x4)
#define RT_MEMORY_SPM ((uint32_t)0x8)
#define RT_MEMORY_P2P_HBM ((uint32_t)0x10)
#define RT_MEMORY_P2P_DDR ((uint32_t)0x11)
#define RT_MEMORY_DDR_NC ((uint32_t)0x20)
#define RT_MEMORY_TS_4G ((uint32_t)0x40)
#define RT_MEMORY_TS ((uint32_t)0x80)
#define RT_MEMORY_RESERVED ((uint32_t)0x100)
#define RT_MEMORY_L1 ((uint32_t)0x1<<16)
#define RT_MEMORY_L2 ((uint32_t)0x1<<17)
typedef uint32_t rtMemType_t;
typedef enum tagRtMemcpyKind {
    RT_MEMCPY_HOST_TO_HOST = 0, RT_MEMCPY_HOST_TO_DEVICE, RT_MEMCPY_DEVICE_TO_HOST,
    RT_MEMCPY_DEVICE_TO_DEVICE, RT_MEMCPY_MANAGED, RT_MEMCPY_ADDR_DEVICE_TO_DEVICE,
    RT_MEMCPY_HOST_TO_DEVICE_EX, RT_MEMCPY_RESERVED
} rtMemcpyKind_t;
RTS_API rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type);
RTS_API rtError_t rtFree(void *devPtr);
RTS_API rtError_t rtMallocHost(void **hostPtr, uint64_t size);
RTS_API rtError_t rtFreeHost(void *hostPtr);
RTS_API rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind);
RTS_API rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count,
                              rtMemcpyKind_t kind, rtStream_t stream);
RTS_API rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
RTS_API rtError_t rtStreamDestroy(rtStream_t stream);
RTS_API rtError_t rtStreamSynchronize(rtStream_t stream);
RTS_API rtError_t rtSetDevice(int32_t device);
RTS_API rtError_t rtDeviceReset(int32_t device);
RTS_API rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId);
RTS_API rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);
#ifdef __cplusplus
}
#endif
#endif
