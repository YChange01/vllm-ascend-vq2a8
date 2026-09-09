// VQ2A8 kernel v2 integration; source provenance: ../vq2a8_expert_reference/README.md.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "acl/acl_rt.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "launch.h"
#include "layout.h"

namespace vq2a8_ascendc_v2 {
namespace {
void CheckTensor(const at::Tensor& tensor, const at::Tensor& x, at::ScalarType dtype, int64_t rank, const char* name,
                 uintptr_t alignment = 32) {
  TORCH_CHECK(tensor.device() == x.device(), name, ": all inputs must share one NPU");
  TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank, name, ": incorrect dtype/rank");
  TORCH_CHECK(tensor.is_contiguous(), name, ": contiguous layout required");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0, name, ": unaligned storage offset");
}

void CheckProjection(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& packed,
                     const at::Tensor& table) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "VQ2A8 v2 requires an NPU tensor");
  CheckTensor(x, x, at::kFloat8_e4m3fn, 2, "activation");
  CheckTensor(scale, x, at::kFloat, 1, "row_scale", 4);
  CheckTensor(bias, x, at::kFloat, 1, "row_bias", 4);
  CheckTensor(packed, x, at::kByte, 4, "packed_zn");
  CheckTensor(table, x, at::kByte, 3, "pair_lut");
  const int64_t m = x.size(0), k = x.size(1), n = packed.size(0) * kN0;
  TORCH_CHECK(ValidDimensions(m, n, k), "VQ2A8 v2 model ABI1 supports M1..32, N4096, K2048/4096 only");
  TORCH_CHECK(scale.numel() == m && bias.numel() == m, "row_scale/row_bias must contain M FP32 values");
  TORCH_CHECK(packed.size(1) == k / kK0 && packed.size(2) == kK0 && packed.size(3) == kN0 / 4,
              "packed_zn shape must be [N/32,K/16,16,8]");
  TORCH_CHECK(table.size(0) == k / kCodebookK && table.size(1) == n / kN0 && table.size(2) == 32,
              "pair_lut shape must be [K/256,N/32,32]");
}

void RecordInputStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) {
  for (const auto& tensor : tensors) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
}  // namespace

std::vector<at::Tensor> GroupedProjection(const std::vector<at::Tensor>& x, const std::vector<at::Tensor>& scale,
                                          const std::vector<at::Tensor>& bias, const std::vector<at::Tensor>& packed,
                                          const std::vector<at::Tensor>& table) {
  const auto jobs = x.size();
  TORCH_CHECK(jobs > 0 && jobs <= kMaxJobs, "VQ2A8 v2 grouped projection requires 1..6 jobs");
  TORCH_CHECK(scale.size() == jobs && bias.size() == jobs && packed.size() == jobs && table.size() == jobs,
              "All VQ2A8 v2 tensor lists must have the same length");
  // Validate every pointer and shape BEFORE descriptor allocation or device work.
  for (size_t i = 0; i < jobs; ++i) {
    CheckProjection(x[i], scale[i], bias[i], packed[i], table[i]);
    TORCH_CHECK(
        x[i].device() == x[0].device() && x[i].size(1) == x[0].size(1) && packed[i].size(0) == packed[0].size(0),
        "VQ2A8 v2 grouped jobs must share device, N and K");
  }
  const c10_npu::OptionalNPUGuard guard(x[0].device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "VQ2A8 v2 model kernel is Ascend950-only");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "CANN platform query failed");
  const uint32_t cores = platform->GetCoreNumAic();
  TORCH_CHECK(cores > 0 && platform->GetCoreNumAiv() >= 2 * cores, "VQ2A8 v2 kernel requires 1C:2V topology");
  const uint32_t n = packed[0].size(0) * kN0;
  const uint32_t nTiles = n / kN;
  const uint32_t blocks = std::min(cores, static_cast<uint32_t>(jobs) * nTiles);
  auto host = at::empty({static_cast<int64_t>(jobs), kJobWords}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto* records = host.data_ptr<int64_t>();
  std::vector<at::Tensor> output;
  output.reserve(jobs);
  for (size_t i = 0; i < jobs; ++i) {
    output.push_back(at::empty({x[i].size(0), n}, x[i].options().dtype(at::kBFloat16)));
    auto* record = records + i * kJobWords;
    record[kX] = reinterpret_cast<int64_t>(x[i].data_ptr());
    record[kScale] = reinterpret_cast<int64_t>(scale[i].data_ptr());
    record[kBias] = reinterpret_cast<int64_t>(bias[i].data_ptr());
    record[kPacked] = reinterpret_cast<int64_t>(packed[i].data_ptr());
    record[kTable] = reinterpret_cast<int64_t>(table[i].data_ptr());
    record[kOutput] = reinterpret_cast<int64_t>(output[i].data_ptr());
    record[kRows] = x[i].size(0);
    record[kColumns] = n;
    record[kReduction] = x[i].size(1);
  }
  // At most 432 metadata bytes H2D. Each expert keeps its independent storage;
  // no weight concatenation, full decoded-weight GM workspace, or model reset.
  // Blocking metadata transfer is intentional and MUST be included in timings.
  auto descriptors = host.to(x[0].device(), at::kLong, false, true);
  const auto stream = c10_npu::getCurrentNPUStream();
  RecordInputStream(x, stream);
  RecordInputStream(scale, stream);
  RecordInputStream(bias, stream);
  RecordInputStream(packed, stream);
  RecordInputStream(table, stream);
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV2GroupedProjection");
  command.SetCustomHandler([stream, blocks, descriptors, jobs, nTiles, x, scale, bias, packed, table, output]() -> int {
    // Retain all tensor owners until the queued launch handler executes. The
    // allocator stream records also protect inputs created on another stream;
    // callers remain responsible for normal producer/consumer stream ordering.
    (void)x;
    (void)scale;
    (void)bias;
    (void)packed;
    (void)table;
    (void)output;
    LaunchGrouped(stream.stream(), blocks, descriptors.data_ptr(), static_cast<uint32_t>(jobs), nTiles);
    return 0;
  });
  command.Run();
  return output;
}
}  // namespace vq2a8_ascendc_v2

TORCH_LIBRARY(vq2a8_ascendc_v2, m) {
  m.def(
      "grouped_projection(Tensor[] x, Tensor[] scale, Tensor[] bias, Tensor[] packed_zn, Tensor[] pair_lut) -> "
      "Tensor[]");
  m.def("abi_version() -> int", []() -> int64_t { return vq2a8_ascendc_v2::kAbiVersion; });
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v2, PrivateUse1, m) {
  m.impl("grouped_projection", &vq2a8_ascendc_v2::GroupedProjection);
}
