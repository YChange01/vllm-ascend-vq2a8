// Internal reference: host transcription restored from chat; see README.md.
// Resident-input kernel batch timing only; not end-to-end model latency or accuracy verification.
#include <acl/acl.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "data_utils.h"
#include "hacl_rt.h"
#include "mat_fp4_tiling.h"
namespace {
static_assert(sizeof(MatFP4TilingData) == 10 * sizeof(uint64_t), "unexpected tiling data layout");
constexpr uint64_t kMaxGroupCount = 128, kKTile = 1024, kZnK0 = 16, kZnN0 = 32;
constexpr uint64_t kCodebookK = 256, kCodebookN = 32, kMxScaleGroupSize = 32;
constexpr uint32_t kKernelCoreCount = 32;
constexpr uint64_t kNAlignment = kKernelCoreCount * kCodebookN;
constexpr uint64_t kNarrowNBlock = 128, kWideNBlock = 256, kN6144 = 6144, kN6144Block = 192;
constexpr uint64_t kN4096SmallM = 96, kN6144SmallM = 64;
constexpr uint64_t kSmallAivK = 512, kSmallAicK = 1024, kSmallMadK = 256, kN6144SmallMadK = 128;
constexpr uint64_t kMiddleAivK = 256, kMiddleAicK = 512, kMiddleMadK = 128;
constexpr uint64_t kLutPrefetchPadding = 7;
constexpr uint64_t kUbCapacity = 256 * 1024, kL1Capacity = 512 * 1024;
constexpr uint64_t kL0ACapacity = 64 * 1024, kL0BCapacity = 64 * 1024, kL0CCapacity = 256 * 1024;
static_assert(kL1Capacity == 2 * kMatFP4L1SlotBytes, "L1 slot layout must cover all L1 memory");
using OutputStorage = uint16_t;
static_assert(sizeof(OutputStorage) == 2, "BF16 output storage must occupy 2 bytes");
constexpr const char *kKernelName = "mat_fp4";
constexpr const char *kKernelBinary = "mat_fp4.o";
struct MatFP4Args {
    uint64_t group_count, k, n;
    void *group_list, *a, *a_scale, *b, *table, *c, *tiling;
};
static_assert(sizeof(MatFP4Args) == 10 * sizeof(uint64_t), "unexpected kernel argument layout");
[[noreturn]] void ThrowError(const std::string &action, int64_t code) {
    throw std::runtime_error(action + " failed, error code=" + std::to_string(code));
}
void CheckRt(rtError_t status, const std::string &action) { if (status != RT_ERROR_NONE) ThrowError(action, status); }
void CheckAcl(aclError status, const std::string &action) { if (status != ACL_SUCCESS) ThrowError(action, status); }
uint64_t CheckedProduct(std::initializer_list<uint64_t> values) {
    uint64_t result = 1;
    for (uint64_t value : values) {
        if (value != 0 && result > std::numeric_limits<uint64_t>::max() / value)
            throw std::overflow_error("tensor byte size overflow");
        result *= value;
    }
    return result;
}
uint64_t CheckedSum(std::initializer_list<uint64_t> values) {
    uint64_t result = 0;
    for (uint64_t value : values) {
        if (result > std::numeric_limits<uint64_t>::max() - value)
            throw std::overflow_error("tensor byte size overflow");
        result += value;
    }
    return result;
}
uint64_t AivUbBytes(uint64_t nbs, uint64_t aiv_kbs) {
    const uint64_t packed_bytes = CheckedProduct({nbs, aiv_kbs}) / 4;
    const uint64_t decoded_bytes = CheckedProduct({nbs, aiv_kbs, kZnK0 + 1}) / kZnK0;
    const uint64_t lut_count = CheckedProduct({aiv_kbs / kCodebookK, nbs / kCodebookN});
    const uint64_t table_bytes = CheckedProduct({CheckedSum({lut_count, kLutPrefetchPadding}), uint64_t{32}});
    return CheckedProduct({uint64_t{2}, CheckedSum({packed_bytes, decoded_bytes, table_bytes})});
}
uint64_t L1SlotBytes(uint64_t m, uint64_t n, uint64_t k) {
    const uint64_t data_bytes = CheckedProduct({m + n, k});
    const uint64_t scale_bytes = CheckedProduct({m, k}) / kMxScaleGroupSize;
    return data_bytes + scale_bytes <= kMatFP4L1SlotBytes ? data_bytes + scale_bytes : data_bytes;
}
bool L1ScaleReusesA(uint64_t m, uint64_t n, uint64_t k) {
    const uint64_t data_bytes = CheckedProduct({m + n, k});
    const uint64_t scale_bytes = CheckedProduct({m, k}) / kMxScaleGroupSize;
    return data_bytes + scale_bytes > kMatFP4L1SlotBytes;
}
uint64_t L1SmallBytes(const MatFP4TilingData &t) {
    return CheckedProduct({uint64_t{2}, L1SlotBytes(t.small_m_limit, t.narrow_nbs, t.small_aic_kbs)});
}
uint64_t L1MiddleBytes(const MatFP4TilingData &t) {
    return CheckedProduct({uint64_t{2}, L1SlotBytes(kMatFP4MaxTileM, t.narrow_nbs, t.middle_aic_kbs)});
}
uint64_t L1WideBytes(const MatFP4TilingData &t) {
    const uint64_t large_m = t.n_only_mode != 0 ? kMatFP4MaxTileM : kMatFP4WideTileM;
    return CheckedProduct({uint64_t{2}, L1SlotBytes(large_m, t.wide_nbs, t.middle_aic_kbs)});
}
uint64_t L1Bytes(const MatFP4TilingData &t) { return std::max({L1SmallBytes(t), L1MiddleBytes(t), L1WideBytes(t)}); }
uint64_t L0ABytes(const MatFP4TilingData &t) {
    const uint64_t tile_bytes = std::max(CheckedProduct({t.small_m_limit, t.small_mad_k}),
                                       CheckedProduct({kMatFP4MaxTileM, t.middle_mad_k}));
    return CheckedProduct({uint64_t{2}, tile_bytes});
}
uint64_t L0BBytes(const MatFP4TilingData &t) {
    const uint64_t narrow_tile_bytes = std::max(CheckedProduct({t.narrow_nbs, t.small_mad_k}),
                                              CheckedProduct({t.narrow_nbs, t.middle_mad_k}));
    const uint64_t tile_bytes = std::max(narrow_tile_bytes, CheckedProduct({t.wide_nbs, t.middle_mad_k}));
    return CheckedProduct({uint64_t{2}, tile_bytes});
}
uint64_t L0CBytes(const MatFP4TilingData &t) {
    const uint64_t double_m = t.n_only_mode != 0 ? t.small_m_limit : kMatFP4MaxTileM;
    const uint64_t large_m = t.n_only_mode != 0 ? kMatFP4MaxTileM : kMatFP4WideTileM;
    const uint64_t narrow_bytes = CheckedProduct({uint64_t{2}, double_m, t.narrow_nbs, sizeof(float)});
    const uint64_t middle_bytes = CheckedProduct({kMatFP4MaxTileM, t.narrow_nbs, sizeof(float)});
    const uint64_t wide_bytes = CheckedProduct({large_m, t.wide_nbs, sizeof(float)});
    return std::max({narrow_bytes, middle_bytes, wide_bytes});
}
uint64_t AivUbMaxBytes(const MatFP4TilingData &t) {
    const uint64_t narrow_bytes = std::max(AivUbBytes(t.narrow_nbs, t.small_aiv_kbs),
                                         AivUbBytes(t.narrow_nbs, t.middle_aiv_kbs));
    return std::max(narrow_bytes, AivUbBytes(t.wide_nbs, t.middle_aiv_kbs));
}
MatFP4TilingData BuildTiling(uint64_t k, uint64_t n) {
    MatFP4TilingData t {};
    const bool n6144 = n == kN6144;
    t.narrow_nbs = n6144 ? kN6144Block : kNarrowNBlock;
    t.wide_nbs = n6144 ? kN6144Block : kWideNBlock;
    t.small_m_limit = n6144 ? kN6144SmallM : kN4096SmallM;
    t.n_only_mode = n6144 ? 1 : 0;
    t.small_aiv_kbs = kSmallAivK; t.small_aic_kbs = kSmallAicK;
    t.small_mad_k = n6144 ? kN6144SmallMadK : kSmallMadK;
    t.middle_aiv_kbs = kMiddleAivK; t.middle_aic_kbs = kMiddleAicK; t.middle_mad_k = kMiddleMadK;
    if (n % t.narrow_nbs != 0 || n % t.wide_nbs != 0)
        throw std::invalid_argument("N must be divisible by the selected N tiles");
    if (k % t.small_aic_kbs != 0 || k % t.middle_aic_kbs != 0)
        throw std::invalid_argument("K must be divisible by both AIC K tiles");
    if (t.small_aic_kbs != 2 * t.small_aiv_kbs || t.middle_aic_kbs != 2 * t.middle_aiv_kbs ||
        t.small_aic_kbs % t.small_mad_k != 0 || t.middle_aic_kbs % t.middle_mad_k != 0 ||
        t.small_aic_kbs % 64 != 0 || t.middle_aic_kbs % 64 != 0)
        throw std::logic_error("invalid fixed AIV/AIC/MMAD tiling");
    const uint64_t large_m = t.n_only_mode != 0 ? kMatFP4MaxTileM : kMatFP4WideTileM;
    if (CheckedProduct({t.small_m_limit + t.narrow_nbs, t.small_aic_kbs}) > kMatFP4L1SlotBytes ||
        CheckedProduct({kMatFP4MaxTileM + t.narrow_nbs, t.middle_aic_kbs}) > kMatFP4L1SlotBytes ||
        CheckedProduct({large_m + t.wide_nbs, t.middle_aic_kbs}) > kMatFP4L1SlotBytes)
        throw std::logic_error("fixed A/B data tiling exceeds one 256 KiB L1 slot");
    if (AivUbMaxBytes(t) > kUbCapacity) throw std::logic_error("fixed tiling exceeds 256 KiB UB capacity");
    if (L1Bytes(t) > kL1Capacity) throw std::logic_error("fixed tiling exceeds 512 KiB L1 capacity");
    if (L0ABytes(t) > kL0ACapacity) throw std::logic_error("fixed tiling exceeds 64 KiB L0A capacity");
    if (L0BBytes(t) > kL0BCapacity) throw std::logic_error("fixed tiling exceeds 64 KiB L0B capacity");
    if (L0CBytes(t) > kL0CCapacity) throw std::logic_error("fixed tiling exceeds 256 KiB L0C capacity");
    return t;
}
size_t ToHostSize(uint64_t size) {
    if (size > std::numeric_limits<size_t>::max()) throw std::overflow_error("tensor is too large for host address space");
    return static_cast<size_t>(size);
}
void ReadExactFile(const std::string &path, void *destination, uint64_t expected_size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open input file: " + path);
    const std::streamoff actual_size = file.tellg();
    if (actual_size < 0 || static_cast<uint64_t>(actual_size) != expected_size)
        throw std::runtime_error(path + " has " + std::to_string(actual_size) + " bytes; expected " +
                                 std::to_string(expected_size));
    file.seekg(0, std::ios::beg);
    file.read(static_cast<char *>(destination), static_cast<std::streamsize>(expected_size));
    if (!file) throw std::runtime_error("failed to read complete input file: " + path);
}
class DeviceBuffer {
public:
    explicit DeviceBuffer(uint64_t bytes) : bytes_(bytes) { CheckRt(rtMalloc(&pointer_, bytes_, RT_MEMORY_HBM), "rtMalloc"); }
    ~DeviceBuffer() { if (pointer_ != nullptr) (void)rtFree(pointer_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void *data() const { return pointer_; }
    uint64_t size() const { return bytes_; }
private:
    void *pointer_ = nullptr;
    uint64_t bytes_ = 0;
};
class RuntimeStream {
public:
    RuntimeStream() { CheckRt(rtStreamCreate(&stream_, 0), "rtStreamCreate"); }
    ~RuntimeStream() { if (stream_ != nullptr) (void)rtStreamDestroy(stream_); }
    RuntimeStream(const RuntimeStream &) = delete;
    RuntimeStream &operator=(const RuntimeStream &) = delete;
    rtStream_t get() const { return stream_; }
    aclrtStream acl_get() const { return reinterpret_cast<aclrtStream>(stream_); }
    void Synchronize(const char *phase) const {
        CheckRt(rtStreamSynchronize(stream_), std::string("rtStreamSynchronize(") + phase + ")");
    }
private:
    rtStream_t stream_ = nullptr;
};
class RuntimeEvent {
public:
    RuntimeEvent() { CheckAcl(aclrtCreateEvent(&event_), "aclrtCreateEvent"); }
    ~RuntimeEvent() { if (event_ != nullptr) (void)aclrtDestroyEvent(event_); }
    RuntimeEvent(const RuntimeEvent &) = delete;
    RuntimeEvent &operator=(const RuntimeEvent &) = delete;
    void Record(const RuntimeStream &stream) const { CheckAcl(aclrtRecordEvent(event_, stream.acl_get()), "aclrtRecordEvent"); }
    float ElapsedMillisecondsTo(const RuntimeEvent &end) const {
        float elapsed_ms = 0.0F;
        CheckAcl(aclrtEventElapsedTime(&elapsed_ms, event_, end.event_), "aclrtEventElapsedTime");
        return elapsed_ms;
    }
private:
    aclrtEvent event_ = nullptr;
};
class RegisteredKernel {
public:
    RegisteredKernel(const char *kernel_name, const char *binary_path) {
        std::ifstream file(binary_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error(std::string("cannot open kernel binary: ") + binary_path);
        const std::streamoff binary_size = file.tellg();
        if (binary_size <= 0) throw std::runtime_error(std::string("empty kernel binary: ") + binary_path);
        binary_.resize(static_cast<size_t>(binary_size));
        file.seekg(0, std::ios::beg);
        file.read(binary_.data(), binary_size);
        if (!file) throw std::runtime_error(std::string("failed to read kernel binary: ") + binary_path);
        rtDevBinary_t desc {};
        desc.magic = RT_DEV_BINARY_MAGIC_ELF; desc.version = 0; desc.data = binary_.data(); desc.length = binary_.size();
        CheckRt(rtDevBinaryRegister(&desc, &binary_handle_), "rtDevBinaryRegister");
        const void *stub = reinterpret_cast<const void *>(kernel_name);
        CheckRt(rtFunctionRegister(binary_handle_, stub, kernel_name, const_cast<void *>(stub), 0), "rtFunctionRegister");
    }
    RegisteredKernel(const RegisteredKernel &) = delete;
    RegisteredKernel &operator=(const RegisteredKernel &) = delete;
private:
    std::vector<char> binary_;
    void *binary_handle_ = nullptr;
};
void ValidateShape(uint64_t group_count, uint64_t k, uint64_t n) {
    if (group_count == 0 || group_count > kMaxGroupCount) throw std::invalid_argument("group count G must be in [1, 128]");
    if (k == 0 || k % kKTile != 0) throw std::invalid_argument("K must be a positive multiple of 1024");
    if (n == 0 || n % kNAlignment != 0) throw std::invalid_argument("N must be a positive multiple of 1024");
}
void CopyHostToDevice(DeviceBuffer &destination, const void *source) {
    CheckRt(rtMemcpy(destination.data(), destination.size(), source, destination.size(), RT_MEMCPY_HOST_TO_DEVICE),
            "rtMemcpy(H2D)");
}
void LaunchGroupPass(const RuntimeStream &stream, uint64_t group_count, uint64_t k, uint64_t n,
    const DeviceBuffer &group_list, const DeviceBuffer &a, const DeviceBuffer &a_scale,
    const DeviceBuffer &all_b, const DeviceBuffer &all_tables, DeviceBuffer &all_c, const DeviceBuffer &tiling) {
    MatFP4Args args {group_count, k, n, group_list.data(), a.data(), a_scale.data(), all_b.data(),
                    all_tables.data(), all_c.data(), tiling.data()};
    CheckRt(rtKernelLaunch(reinterpret_cast<const void *>(kKernelName), kKernelCoreCount, &args, sizeof(args),
                           nullptr, stream.get()), "rtKernelLaunch(serial group pass)");
}
void RunTest(uint64_t group_count, uint64_t k, uint64_t n, uint64_t warmup_cycles, uint64_t test_cycles) {
    ValidateShape(group_count, k, n);
    if (test_cycles == 0) throw std::invalid_argument("test_cycles must be at least 1");
    const uint64_t group_list_bytes = CheckedProduct({group_count, sizeof(int64_t)});
    std::vector<int64_t> host_group_list(ToHostSize(group_count));
    ReadExactFile("input/input_group_list.bin", host_group_list.data(), group_list_bytes);
    uint64_t total_m = 0, active_group_count = 0;
    for (uint64_t group_id = 0; group_id < group_count; ++group_id) {
        const int64_t signed_group_m = host_group_list[ToHostSize(group_id)];
        if (signed_group_m < 0) throw std::invalid_argument("every group_list[i] must be non-negative; group_list[" +
            std::to_string(group_id) + "]=" + std::to_string(signed_group_m));
        const uint64_t group_m = static_cast<uint64_t>(signed_group_m);
        if (total_m > std::numeric_limits<uint64_t>::max() - group_m) throw std::overflow_error("sum(group_list) overflow");
        total_m += group_m; active_group_count += group_m != 0;
    }
    if (total_m == 0) throw std::invalid_argument("group_list must contain at least one token row");
    const MatFP4TilingData tiling = BuildTiling(k, n);
    const uint64_t a_bytes = CheckedProduct({total_m, k});
    const uint64_t a_scale_bytes = CheckedProduct({total_m, k}) / kMxScaleGroupSize;
    const uint64_t b_group_bytes = CheckedProduct({n, k}) / 4;
    const uint64_t table_group_bytes = CheckedProduct({k / kCodebookK, n / kCodebookN, uint64_t{32}});
    const uint64_t all_b_bytes = CheckedProduct({group_count, b_group_bytes});
    const uint64_t all_table_bytes = CheckedProduct({group_count, table_group_bytes});
    const uint64_t all_c_elements = CheckedProduct({total_m, n});
    const uint64_t all_c_bytes = CheckedProduct({all_c_elements, sizeof(OutputStorage)});
    std::vector<uint8_t> host_a(ToHostSize(a_bytes)), host_a_scale(ToHostSize(a_scale_bytes));
    std::vector<uint8_t> host_b(ToHostSize(all_b_bytes)), host_tables(ToHostSize(all_table_bytes));
    std::vector<OutputStorage> host_c(ToHostSize(all_c_elements), 0);
    ReadExactFile("input/input_a.bin", host_a.data(), a_bytes);
    ReadExactFile("input/input_a_scale.bin", host_a_scale.data(), a_scale_bytes);
    ReadExactFile("input/input_b.bin", host_b.data(), all_b_bytes);
    ReadExactFile("input/input_table.bin", host_tables.data(), all_table_bytes);
    RegisteredKernel registered_kernel(kKernelName, kKernelBinary);
    DeviceBuffer device_group_list(group_list_bytes), device_a(a_bytes), device_a_scale(a_scale_bytes);
    DeviceBuffer device_b(all_b_bytes), device_tables(all_table_bytes), device_c(all_c_bytes), device_tiling(sizeof(tiling));
    RuntimeStream stream;
    CopyHostToDevice(device_group_list, host_group_list.data());
    CopyHostToDevice(device_a, host_a.data()); CopyHostToDevice(device_a_scale, host_a_scale.data());
    CopyHostToDevice(device_b, host_b.data()); CopyHostToDevice(device_tables, host_tables.data());
    CopyHostToDevice(device_c, host_c.data()); CopyHostToDevice(device_tiling, &tiling);
    for (uint64_t cycle = 0; cycle < warmup_cycles; ++cycle)
        LaunchGroupPass(stream, group_count, k, n, device_group_list, device_a, device_a_scale,
                        device_b, device_tables, device_c, device_tiling);
    stream.Synchronize("after warmup kernels");
    RuntimeEvent start_event, end_event;
    // Inputs stay resident. Time the complete launch batch once, then divide by its launch count.
    // Recording the same pair inside this loop would retain only the final interval, not a sum.
    start_event.Record(stream);
    for (uint64_t cycle = 0; cycle < test_cycles; ++cycle) {
        LaunchGroupPass(stream, group_count, k, n, device_group_list, device_a, device_a_scale,
                        device_b, device_tables, device_c, device_tiling);
    }
    end_event.Record(stream);
    stream.Synchronize("after timed kernels");
    const double elapsed_ms = static_cast<double>(start_event.ElapsedMillisecondsTo(end_event));
    if (!std::isfinite(elapsed_ms) || elapsed_ms <= 0.0)
        throw std::runtime_error("resident kernel batch timing must be finite and positive; increase test_cycles");
    const double pass_ms = elapsed_ms / static_cast<double>(test_cycles);
    CheckRt(rtMemcpy(host_c.data(), all_c_bytes, device_c.data(), all_c_bytes, RT_MEMCPY_DEVICE_TO_HOST), "rtMemcpy(D2H output)");
    if (mkdir("output", 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create output directory: " + std::string(std::strerror(errno)));
    if (!WriteFile("output/output_c.bin", host_c.data(), all_c_bytes))
        throw std::runtime_error("failed to write output/output_c.bin");
    const double group_ms = pass_ms / static_cast<double>(active_group_count);
    const double operations = 2.0 * static_cast<double>(total_m) * static_cast<double>(n) * static_cast<double>(k);
    const double throughput_tflops = operations / (pass_ms * 1.0e9);
    std::string group_list_text = "[";
    for (uint64_t group_id = 0; group_id < group_count; ++group_id) {
        if (group_id != 0) group_list_text += ", ";
        group_list_text += std::to_string(host_group_list[ToHostSize(group_id)]);
    }
    group_list_text += "]";
    // Reporting compacted during transcription; timing corrected to one resident batch interval.
    std::cout << "W2A8 grouped matmul run completed\n"
              << "  group_list: " << group_list_text << " (non-cumulative M_i)\n"
              << "  group count: " << group_count << " active: " << active_group_count << " rows: " << total_m << '\n'
              << "  A shape: [" << total_m << "," << k << "] MXFP8 E4M3\n"
              << "  A-scale shape: [" << total_m << "," << k/64 << ",2] E8M0\n"
              << "  B packed zN shape: [" << group_count << "," << n/kZnN0 << "," << k/kZnK0 << ",16,8]\n"
              << "  pair LUT shape: [" << group_count << "," << k/kCodebookK << "," << n/kCodebookN << ",32] bytes\n"
              << "  A layout: GM ND -> L1 zN -> L0A; scale GM DN -> L1 NZ -> L0A_MX\n"
              << "  B layout: GM zN -> UB zN -> L1 zN -> L0B; resident E8M0(1) L0B_MX\n"
              << "  core policy: " << (tiling.n_only_mode != 0 ? "1x32 N-only\n" : "1x32 M<480; 2x16 M>=480\n")
              << "  small tile M/N/K: " << tiling.small_m_limit << '/' << tiling.narrow_nbs << '/' << tiling.small_aic_kbs
              << " AIV K=" << tiling.small_aiv_kbs << " MAD K=" << tiling.small_mad_k << '\n'
              << "  middle tile M/N/K: 256/" << tiling.narrow_nbs << '/' << tiling.middle_aic_kbs
              << " AIV K=" << tiling.middle_aiv_kbs << " MAD K=" << tiling.middle_mad_k << '\n'
              << "  large tile M/N/K: " << (tiling.n_only_mode ? kMatFP4MaxTileM : kMatFP4WideTileM)
              << '/' << tiling.wide_nbs << '/' << tiling.middle_aic_kbs << '\n'
              << "  AIV UB bytes: " << AivUbMaxBytes(tiling) << '/' << kUbCapacity << '\n'
              << "  L1 small/middle/wide/max bytes: " << L1SmallBytes(tiling) << '/' << L1MiddleBytes(tiling)
              << '/' << L1WideBytes(tiling) << '/' << L1Bytes(tiling) << " capacity=" << kL1Capacity << '\n'
              << "  A-scale L1 policy: " << (L1ScaleReusesA(tiling.small_m_limit, tiling.narrow_nbs, tiling.small_aic_kbs)
                      ? "small uses A-tail or consumed A scratch\n" : "resident after B\n")
              << "  C shape: [" << total_m << ',' << n << "] BF16\n"
              << "  physical core launch: " << kKernelCoreCount << '\n'
              << "  timed group passes: " << test_cycles << " kernels per pass: 1\n"
              << "  timing source: device events across one complete resident-input launch batch\n"
              << "  timing scope: resident kernel batch average; excludes H2D/D2H and model preparation, not E2E\n"
              << "  accuracy verification: NOT PERFORMED (output file only)\n"
              << "  batch elapsed: " << elapsed_ms << " ms\n"
              << "  average serial pass: " << pass_ms << " ms\n"
              << "  amortized active group: " << group_ms << " ms (not per-group measured latency)\n"
              << "  aggregate throughput: " << throughput_tflops << " TFLOP/s\n";
}
uint64_t ParseUnsigned(const char *text, const char *name) {
    try {
        size_t parsed = 0;
        const std::string value(text);
        if (value.empty() || value.front() == '-') throw std::invalid_argument("negative or empty value");
        const uint64_t result = std::stoull(value, &parsed);
        if (parsed != value.size()) throw std::invalid_argument("trailing characters");
        return result;
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
}
void PrintUsage(const char *program) {
    std::cerr << "Usage: " << program << " <G=groups,1..128> <K> <N> [device_id=0] [warmup_cycles=0] [test_cycles=1]\n";
}
} // namespace
int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 7) { PrintUsage(argv[0]); return 2; }
    bool acl_initialized = false, device_selected = false;
    int32_t device_id = 0;
    try {
        const uint64_t group_count = ParseUnsigned(argv[1], "G");
        const uint64_t k = ParseUnsigned(argv[2], "K"), n = ParseUnsigned(argv[3], "N");
        const uint64_t raw_device_id = argc >= 5 ? ParseUnsigned(argv[4], "device_id") : 0;
        if (raw_device_id > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument("device_id is outside int32 range");
        device_id = static_cast<int32_t>(raw_device_id);
        const uint64_t warmup_cycles = argc >= 6 ? ParseUnsigned(argv[5], "warmup_cycles") : 0;
        const uint64_t test_cycles = argc >= 7 ? ParseUnsigned(argv[6], "test_cycles") : 1;
        CheckAcl(aclInit(nullptr), "aclInit"); acl_initialized = true;
        CheckAcl(aclrtSetDevice(device_id), "aclrtSetDevice"); device_selected = true;
        RunTest(group_count, k, n, warmup_cycles, test_cycles);
        CheckAcl(aclrtResetDevice(device_id), "aclrtResetDevice"); device_selected = false;
        CheckAcl(aclFinalize(), "aclFinalize"); acl_initialized = false;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        if (device_selected) (void)aclrtResetDevice(device_id);
        if (acl_initialized) (void)aclFinalize();
        return 1;
    }
}
