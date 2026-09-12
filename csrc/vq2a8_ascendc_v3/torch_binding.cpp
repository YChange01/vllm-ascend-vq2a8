// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#include "prepare_launch.h"
#include "resident_launch.h"
#include "resident_layout.h"

namespace vq2a8_v3 {
namespace {
void RecordInputStream(const std::vector<at::Tensor>& tensors, c10_npu::NPUStream stream) {
  for (const auto& tensor : tensors) {
    if (tensor.defined()) c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
  }
}

void Check(const at::Tensor& t, const at::Tensor& x, at::ScalarType dtype, int64_t rank, const char* name,
           uintptr_t alignment = 32) {
  TORCH_CHECK(t.device() == x.device(), name, ": all tensors must be on the same NPU");
  TORCH_CHECK(t.scalar_type() == dtype && t.dim() == rank, name, ": wrong dtype/rank");
  TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment == 0, name, ": unaligned storage offset");
}

void CheckX(const at::Tensor& x) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "AscendC prototype requires an NPU");
  Check(x, x, at::kFloat8_e4m3fn, 2, "activation");
  TORCH_CHECK(ValidDimensions(x.size(0), 32, x.size(1), 1), "Require 1<=M<=32, K%512=0, K<=65536");
}

at::Tensor Run(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& packed,
               const at::Tensor& book, const at::Tensor& ids, const at::Tensor& dense, uint32_t n, uint32_t tiles,
               uint32_t mode) {
  const c10_npu::OptionalNPUGuard guard(x.device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "AscendC VQ2A8 prototype is Ascend950-only");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "CANN platform query failed");
  uint32_t cores = platform->GetCoreNumAic();
  TORCH_CHECK(cores > 0 && platform->GetCoreNumAiv() >= 2 * cores, "Require a 1C:2V core topology");
  uint32_t blocks = std::min(cores, n / kN);
  auto output = at::empty({x.size(0), n}, x.options().dtype(at::kBFloat16));
  auto npuStream = c10_npu::getCurrentNPUStream();
  RecordInputStream({x, scale, bias, packed, book, ids, dense}, npuStream);
  auto stream = npuStream.stream();
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV3Projection");
  // Retain Tensor owners through the custom handler. No host staging,
  // synchronizations or value scans on the projection hot path.
  command.SetCustomHandler([=]() -> int {
    Launch(stream, blocks, x.data_ptr(), scale.data_ptr(), bias.data_ptr(),
           packed.defined() ? packed.data_ptr() : nullptr, book.defined() ? book.data_ptr() : nullptr,
           ids.defined() ? ids.data_ptr() : nullptr, dense.defined() ? dense.data_ptr() : nullptr, output.data_ptr(),
           x.size(0), n, x.size(1), tiles, mode);
    return 0;
  });
  command.Run();
  return output;
}
}  // namespace

void CheckProjection(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& packed,
                     const at::Tensor& book, const at::Tensor& ids) {
  CheckX(x);
  Check(scale, x, at::kFloat, 1, "scale", 4);
  Check(bias, x, at::kFloat, 1, "bias", 4);
  Check(packed, x, at::kInt, 2, "packed_indices");
  Check(book, x, at::kFloat8_e4m3fn, 4, "codebooks");
  Check(ids, x, at::kByte, 1, "tile_ids");
  auto m = x.size(0), k = x.size(1), n = packed.size(0) * 2, tiles = book.size(0);
  TORCH_CHECK(ValidDimensions(m, n, k, tiles), "Unsupported VQ2A8 prototype dimensions");
  TORCH_CHECK(scale.numel() == m && bias.numel() == m, "scale/bias must contain M values");
  TORCH_CHECK(packed.size(1) == k / 8 && ids.numel() == k, "packed/IDs must cover K exactly");
  TORCH_CHECK(book.size(1) == n / 32 && book.size(2) == 16 && book.size(3) == 2,
              "codebooks must have shape [tiles,N/32,16,2]");
}

at::Tensor Projection(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& packed,
                      const at::Tensor& book, const at::Tensor& ids) {
  CheckProjection(x, scale, bias, packed, book, ids);
  return Run(x, scale, bias, packed, book, ids, {}, packed.size(0) * 2, book.size(0), 2);
}

template <bool Pipeline = false>
std::vector<at::Tensor> GroupedProjection(const std::vector<at::Tensor>& x, const std::vector<at::Tensor>& scale,
                                          const std::vector<at::Tensor>& bias, const std::vector<at::Tensor>& packed,
                                          const std::vector<at::Tensor>& book, const std::vector<at::Tensor>& ids) {
  auto jobs = x.size();
  TORCH_CHECK(jobs > 0 && jobs <= kMaxJobs, "Grouped projection requires 1..6 jobs");
  TORCH_CHECK(
      scale.size() == jobs && bias.size() == jobs && packed.size() == jobs && book.size() == jobs && ids.size() == jobs,
      "Grouped tensor lists must have identical lengths");
  // Validate ALL jobs before allocating descriptors or launching anything.
  for (size_t i = 0; i < jobs; ++i) {
    CheckProjection(x[i], scale[i], bias[i], packed[i], book[i], ids[i]);
    TORCH_CHECK(
        x[i].device() == x[0].device() && x[i].size(1) == x[0].size(1) && packed[i].size(0) == packed[0].size(0),
        "Grouped jobs must share device, N and K");
  }
  const c10_npu::OptionalNPUGuard guard(x[0].device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "Grouped VQ2A8 is Ascend950-only");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "CANN platform query failed");
  uint32_t cores = platform->GetCoreNumAic();
  TORCH_CHECK(cores > 0 && platform->GetCoreNumAiv() >= 2 * cores, "Require a 1C:2V core topology");
  uint32_t n = packed[0].size(0) * 2, groups = n / kN;
  uint32_t blocks = std::min(cores, static_cast<uint32_t>(jobs) * groups);
  auto host = at::zeros({static_cast<int64_t>(jobs), kJobWords}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto* records = host.data_ptr<int64_t>();
  std::vector<at::Tensor> output;
  for (size_t i = 0; i < jobs; ++i) {
    output.push_back(at::empty({x[i].size(0), n}, x[i].options().dtype(at::kBFloat16)));
    auto* entry = records + i * kJobWords;
    entry[0] = reinterpret_cast<int64_t>(x[i].data_ptr());
    entry[1] = reinterpret_cast<int64_t>(scale[i].data_ptr());
    entry[2] = reinterpret_cast<int64_t>(bias[i].data_ptr());
    entry[3] = reinterpret_cast<int64_t>(packed[i].data_ptr());
    entry[4] = reinterpret_cast<int64_t>(book[i].data_ptr());
    entry[5] = reinterpret_cast<int64_t>(ids[i].data_ptr());
    entry[6] = reinterpret_cast<int64_t>(output[i].data_ptr());
    entry[7] = x[i].size(0);
    entry[8] = n;
    entry[9] = x[i].size(1);
    entry[10] = book[i].size(0);
  }
  // Only <=576 bytes of launch metadata cross H2D, never expert weights.
  // Blocking copy keeps host lifetime unambiguous; included in grouped timing.
  auto descriptors = host.to(x[0].device(), at::kLong, false, true);
  auto npuStream = c10_npu::getCurrentNPUStream();
  RecordInputStream(x, npuStream);
  RecordInputStream(scale, npuStream);
  RecordInputStream(bias, npuStream);
  RecordInputStream(packed, npuStream);
  RecordInputStream(book, npuStream);
  RecordInputStream(ids, npuStream);
  auto stream = npuStream.stream();
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV3GroupedProjection");
  command.SetCustomHandler(
      [stream, blocks, descriptors, jobs, groups, x, scale, bias, packed, book, ids, output]() -> int {
        // Explicitly retain EVERY pointer owner, even though Launch takes only the
        // descriptor address. Same-stream allocator reuse is ordered after launch.
        (void)x;
        (void)scale;
        (void)bias;
        (void)packed;
        (void)book;
        (void)ids;
        (void)output;
        if constexpr (Pipeline) {
          LaunchGroupedPipeline(stream, blocks, descriptors.data_ptr(), static_cast<uint32_t>(jobs), groups);
        } else {
          LaunchGrouped(stream, blocks, descriptors.data_ptr(), static_cast<uint32_t>(jobs), groups);
        }
        return 0;
      });
  command.Run();
  return output;
}

at::Tensor CubeControl(const at::Tensor& x, const at::Tensor& w, bool bridge) {
  CheckX(x);
  Check(w, x, at::kFloat8_e4m3fn, 2, "synthetic_weight");
  TORCH_CHECK(w.size(1) == x.size(1) && ValidDimensions(x.size(0), w.size(0), x.size(1), 1),
              "Control weight must have shape [N,K], N%32=0");
  // Bounded synthetic matrices only. Real experts must use Projection.
  TORCH_CHECK(w.numel() <= 64 * 1024, "Dense control is restricted to <=65536 synthetic weight bytes");
  auto options = x.options().dtype(at::kFloat);
  return Run(x, at::ones({x.size(0)}, options), at::zeros({x.size(0)}, options), {}, {}, {}, w, w.size(0), 1,
             bridge ? 1 : 0);
}

at::Tensor MakeConstants(const at::Tensor& anchor) {
  TORCH_CHECK(anchor.device().type() == c10::DeviceType::PrivateUse1, "V3 constants require an NPU");
  const c10_npu::OptionalNPUGuard guard(anchor.device());
  auto host = at::empty({kConstantWords}, at::TensorOptions().device(at::kCPU).dtype(at::kInt));
  FillConstantWords(reinterpret_cast<uint32_t*>(host.data_ptr<int32_t>()));
  // Initialization only. The workspace owns this immutable allocation for all
  // subsequent launches; the projection operation below never stages metadata.
  return host.to(anchor.device(), at::kInt, false, true);
}

void GroupedProjectionOut(const at::Tensor& descriptors, const at::Tensor& constants,
                          const std::vector<at::Tensor>& owners, int64_t jobs, int64_t m, int64_t n, int64_t k,
                          int64_t tiles, bool pipeline) {
  // Internal trusted-workspace ABI. Descriptor pointers are created only by the
  // pinned Python workspace from validated, immutable resident banks. Do not
  // accept descriptor bytes from files or RPC clients. Reading them back here
  // would reintroduce the per-layer device/host synchronization v3 removes.
  TORCH_CHECK(descriptors.device().type() == c10::DeviceType::PrivateUse1, "V3 requires an NPU");
  TORCH_CHECK(jobs > 0 && jobs <= kMaxJobs && m == 1, "Resident V3 decode requires 1..6 M=1 jobs");
  TORCH_CHECK(n > 0 && n <= 65536 && k > 0 && k <= 65536 && tiles > 0 && tiles <= kMaxTiles,
              "Invalid V3 projection bounds");
  TORCH_CHECK(ValidDimensions(m, n, k, tiles), "Unsupported V3 dimensions");
  Check(descriptors, descriptors, at::kLong, 2, "descriptors");
  Check(constants, descriptors, at::kInt, 1, "constants");
  TORCH_CHECK(descriptors.size(0) == jobs && descriptors.size(1) == kJobWords, "V3 descriptors must be [jobs,12]");
  TORCH_CHECK(constants.numel() == kConstantWords, "Wrong V3 constant-table ABI");
  TORCH_CHECK(!owners.empty(), "V3 workspace must retain all pointer owners");
  for (const auto& owner : owners) {
    TORCH_CHECK(owner.defined() && owner.device() == descriptors.device(), "V3 owners must share the NPU");
  }
  const c10_npu::OptionalNPUGuard guard(descriptors.device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "V3 is Ascend950-only");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "CANN platform query failed");
  const uint32_t cores = platform->GetCoreNumAic();
  TORCH_CHECK(cores > 0 && platform->GetCoreNumAiv() >= 2 * cores, "Require a 1C:2V core topology");
  const uint32_t groups = static_cast<uint32_t>(n) / kN;
  const uint32_t blocks = std::min(cores, static_cast<uint32_t>(jobs) * groups);
  auto npuStream = c10_npu::getCurrentNPUStream();
  RecordInputStream(owners, npuStream);
  RecordInputStream({descriptors, constants}, npuStream);
  auto stream = npuStream.stream();
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV3ResidentProjection");
  command.SetCustomHandler([stream, blocks, descriptors, constants, owners, jobs, groups, pipeline]() -> int {
    // Same-stream workspace reuse is ordered. Strong owners survive deferred
    // host dispatch. Python forbids concurrent/cross-stream workspace use and
    // allocator stream records delay reuse until device consumers complete,
    // including early release on the host after this handler has enqueued.
    (void)owners;
    LaunchGroupedV3(stream, blocks, descriptors.data_ptr(), constants.data_ptr(), static_cast<uint32_t>(jobs), groups,
                    pipeline);
    return 0;
  });
  command.Run();
}

void CheckResidentProjection(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                             const at::Tensor& packed, const at::Tensor& table) {
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "V3 resident projection requires an NPU");
  Check(x, x, at::kFloat8_e4m3fn, 2, "activation");
  Check(scale, x, at::kFloat, 1, "row_scale", 4);
  Check(bias, x, at::kFloat, 1, "row_bias", 4);
  Check(packed, x, at::kByte, 4, "packed_zn");
  Check(table, x, at::kByte, 3, "pair_lut");
  const int64_t m = x.size(0), k = x.size(1), n = packed.size(0) * vq2a8_v3_resident::kN0;
  TORCH_CHECK(vq2a8_v3_resident::ValidDimensions(m, n, k),
              "V3 resident projection supports M1..32, N4096, K2048/4096 only");
  TORCH_CHECK(scale.numel() == m && bias.numel() == m, "row_scale/row_bias must contain M FP32 values");
  TORCH_CHECK(packed.size(1) == k / vq2a8_v3_resident::kK0 && packed.size(2) == vq2a8_v3_resident::kK0 &&
                  packed.size(3) == vq2a8_v3_resident::kN0 / 4,
              "packed_zn must have shape [N/32,K/16,16,8]");
  TORCH_CHECK(table.size(0) == k / vq2a8_v3_resident::kCodebookK && table.size(1) == n / vq2a8_v3_resident::kN0 &&
                  table.size(2) == 32,
              "pair_lut must have shape [K/256,N/32,32]");
}

uint32_t ResidentCoreCount() {
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "V3 resident projection is Ascend950-only");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "CANN platform query failed");
  const uint32_t cores = platform->GetCoreNumAic();
  TORCH_CHECK(cores > 0 && platform->GetCoreNumAiv() >= 2 * cores, "V3 resident projection requires 1C:2V topology");
  return cores;
}

std::vector<at::Tensor> GroupedProjectionResident(const std::vector<at::Tensor>& x,
                                                  const std::vector<at::Tensor>& scale,
                                                  const std::vector<at::Tensor>& bias,
                                                  const std::vector<at::Tensor>& packed,
                                                  const std::vector<at::Tensor>& table) {
  const auto jobs = x.size();
  TORCH_CHECK(jobs > 0 && jobs <= vq2a8_v3_resident::kMaxJobs, "V3 resident projection requires 1..6 jobs");
  TORCH_CHECK(scale.size() == jobs && bias.size() == jobs && packed.size() == jobs && table.size() == jobs,
              "V3 resident tensor lists must have identical lengths");
  // Eager prefill/preflight entry. Validate every pointer before allocating or
  // submitting device work; decode uses the prepared out entry below instead.
  for (size_t i = 0; i < jobs; ++i) {
    CheckResidentProjection(x[i], scale[i], bias[i], packed[i], table[i]);
    TORCH_CHECK(
        x[i].device() == x[0].device() && x[i].size(1) == x[0].size(1) && packed[i].size(0) == packed[0].size(0),
        "V3 resident grouped jobs must share device, N and K");
  }
  const c10_npu::OptionalNPUGuard guard(x[0].device());
  const uint32_t cores = ResidentCoreCount();
  const uint32_t n = packed[0].size(0) * vq2a8_v3_resident::kN0;
  const uint32_t groups = n / vq2a8_v3_resident::kN;
  const uint32_t blocks = std::min(cores, static_cast<uint32_t>(jobs) * groups);
  auto host = at::empty({static_cast<int64_t>(jobs), vq2a8_v3_resident::kJobWords},
                        at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto* records = host.data_ptr<int64_t>();
  std::vector<at::Tensor> output;
  output.reserve(jobs);
  for (size_t i = 0; i < jobs; ++i) {
    output.push_back(at::empty({x[i].size(0), n}, x[i].options().dtype(at::kBFloat16)));
    auto* entry = records + i * vq2a8_v3_resident::kJobWords;
    entry[vq2a8_v3_resident::kX] = reinterpret_cast<int64_t>(x[i].data_ptr());
    entry[vq2a8_v3_resident::kScale] = reinterpret_cast<int64_t>(scale[i].data_ptr());
    entry[vq2a8_v3_resident::kBias] = reinterpret_cast<int64_t>(bias[i].data_ptr());
    entry[vq2a8_v3_resident::kPacked] = reinterpret_cast<int64_t>(packed[i].data_ptr());
    entry[vq2a8_v3_resident::kTable] = reinterpret_cast<int64_t>(table[i].data_ptr());
    entry[vq2a8_v3_resident::kOutput] = reinterpret_cast<int64_t>(output[i].data_ptr());
    entry[vq2a8_v3_resident::kRows] = x[i].size(0);
    entry[vq2a8_v3_resident::kColumns] = n;
    entry[vq2a8_v3_resident::kReduction] = x[i].size(1);
  }
  // This eager-only blocking metadata copy must be included in its timings.
  auto descriptors = host.to(x[0].device(), at::kLong, false, true);
  const auto npuStream = c10_npu::getCurrentNPUStream();
  RecordInputStream(x, npuStream);
  RecordInputStream(scale, npuStream);
  RecordInputStream(bias, npuStream);
  RecordInputStream(packed, npuStream);
  RecordInputStream(table, npuStream);
  const auto stream = npuStream.stream();
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV3ResidentGroupedProjection");
  command.SetCustomHandler([stream, blocks, descriptors, jobs, groups, x, scale, bias, packed, table, output]() -> int {
    (void)x;
    (void)scale;
    (void)bias;
    (void)packed;
    (void)table;
    (void)output;
    vq2a8_v3_resident::LaunchGrouped(stream, blocks, descriptors.data_ptr(), static_cast<uint32_t>(jobs), groups);
    return 0;
  });
  command.Run();
  return output;
}

void GroupedProjectionResidentOut(const at::Tensor& descriptors, const std::vector<at::Tensor>& owners, int64_t jobs,
                                  int64_t m, int64_t n, int64_t k) {
  // Trusted workspace ABI 1: int64[jobs,9] contains x, scale, bias, packed_zn,
  // pair_lut, output, M, N, K. Every pointed allocation belongs to owners.
  // Python constructs validated immutable banks and updates selected pointers
  // on the same stream. Never inspect descriptor values through a host copy.
  TORCH_CHECK(descriptors.device().type() == c10::DeviceType::PrivateUse1, "V3 resident projection requires an NPU");
  TORCH_CHECK(jobs > 0 && jobs <= vq2a8_v3_resident::kMaxJobs && m == 1, "V3 resident decode requires 1..6 M1 jobs");
  TORCH_CHECK(vq2a8_v3_resident::ValidDimensions(m, n, k), "Unsupported V3 resident projection dimensions");
  Check(descriptors, descriptors, at::kLong, 2, "resident_descriptors");
  TORCH_CHECK(descriptors.size(0) == jobs && descriptors.size(1) == vq2a8_v3_resident::kJobWords,
              "V3 resident descriptors must have shape [jobs,9]");
  TORCH_CHECK(!owners.empty(), "V3 resident workspace must retain every pointer owner");
  for (const auto& owner : owners) {
    TORCH_CHECK(owner.defined() && owner.device() == descriptors.device(), "V3 resident owners must share the NPU");
  }
  const c10_npu::OptionalNPUGuard guard(descriptors.device());
  const uint32_t cores = ResidentCoreCount();
  const uint32_t groups = static_cast<uint32_t>(n) / vq2a8_v3_resident::kN;
  const uint32_t blocks = std::min(cores, static_cast<uint32_t>(jobs) * groups);
  const auto npuStream = c10_npu::getCurrentNPUStream();
  RecordInputStream(owners, npuStream);
  RecordInputStream({descriptors}, npuStream);
  const auto stream = npuStream.stream();
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV3ResidentProjectionOut");
  command.SetCustomHandler([stream, blocks, descriptors, owners, jobs, groups]() -> int {
    // Retain owners through deferred host dispatch and record their stream use
    // above so early host release cannot recycle live device allocations.
    (void)owners;
    vq2a8_v3_resident::LaunchGrouped(stream, blocks, descriptors.data_ptr(), static_cast<uint32_t>(jobs), groups);
    return 0;
  });
  command.Run();
}

void PrepareOut(const at::Tensor& rotated, const at::Tensor& weightScale, const at::Tensor& order,
                const at::Tensor& inputBias, const at::Tensor& quantized, const at::Tensor& scale,
                const at::Tensor& bias, const at::Tensor& valid) {
  TORCH_CHECK(rotated.device().type() == c10::DeviceType::PrivateUse1, "V3 preparation requires an NPU");
  Check(rotated, rotated, at::kFloat, 2, "rotated");
  Check(weightScale, rotated, at::kFloat, 2, "weight_scale");
  Check(order, rotated, at::kLong, 2, "activation_order");
  Check(inputBias, rotated, at::kFloat, 1, "input_bias", 4);
  Check(quantized, rotated, at::kFloat8_e4m3fn, 2, "quantized");
  Check(scale, rotated, at::kFloat, 1, "row_scale", 4);
  Check(bias, rotated, at::kFloat, 1, "output_bias", 4);
  Check(valid, rotated, at::kInt, 1, "valid", 4);
  const int64_t jobs = rotated.size(0), k = rotated.size(1);
  TORCH_CHECK(ValidPrepareDimensions(jobs, k), "V3 preparation requires 1..6 rows and K%512=0, K<=65536");
  TORCH_CHECK(weightScale.sizes() == rotated.sizes() && order.sizes() == rotated.sizes() &&
                  quantized.sizes() == rotated.sizes(),
              "V3 preparation inputs and quantized output must have the same [jobs,K] shape");
  TORCH_CHECK(inputBias.numel() == jobs && scale.numel() == jobs && bias.numel() == jobs && valid.numel() == jobs,
              "V3 preparation bias/scale/valid tensors must contain one value per row");
  const std::vector<at::Tensor> owners{rotated, weightScale, order, inputBias, quantized, scale, bias, valid};
  constexpr size_t kFirstOutput = 4;
  for (size_t output = kFirstOutput; output < owners.size(); ++output) {
    for (size_t other = 0; other < output; ++other) {
      // Reject even disjoint views of the same storage: asynchronous row
      // processing must never overwrite another row's input or output.
      TORCH_CHECK(!owners[output].is_alias_of(owners[other]), "V3 preparation outputs require distinct storage");
    }
  }
  const c10_npu::OptionalNPUGuard guard(rotated.device());
  const char* soc = aclrtGetSocName();
  TORCH_CHECK(soc && std::strncmp(soc, "Ascend950", 9) == 0, "V3 preparation is Ascend950-only");
  auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  TORCH_CHECK(platform != nullptr, "CANN platform query failed");
  const uint32_t cores = platform->GetCoreNumAiv();
  TORCH_CHECK(cores > 0, "V3 preparation requires vector cores");
  const uint32_t blocks = std::min(cores, static_cast<uint32_t>(jobs));
  const auto npuStream = c10_npu::getCurrentNPUStream();
  RecordInputStream(owners, npuStream);
  const auto stream = npuStream.stream();
  at_npu::native::OpCommand command;
  command.Name("Vq2a8AscendCV3PrepareOut");
  command.SetCustomHandler([stream, blocks, owners, jobs, k]() -> int {
    LaunchPrepareV3(stream, blocks, owners[0].data_ptr(), owners[1].data_ptr(), owners[2].data_ptr(),
                    owners[3].data_ptr(), owners[4].data_ptr(), owners[5].data_ptr(), owners[6].data_ptr(),
                    owners[7].data_ptr(), static_cast<uint32_t>(jobs), static_cast<uint32_t>(k));
    return 0;
  });
  command.Run();
}
}  // namespace vq2a8_v3

TORCH_LIBRARY(vq2a8_ascendc_v3, m) {
  m.def("resident_abi_version() -> int", []() -> int64_t { return vq2a8_v3_resident::kAbiVersion; });
  m.def("resident_capabilities() -> int", []() -> int64_t { return vq2a8_v3_resident::kCapabilities; });
  m.def(
      "prepare_out(Tensor rotated, Tensor weight_scale, Tensor order, Tensor input_bias, Tensor(a!) quantized, "
      "Tensor(b!) scale, Tensor(c!) bias, Tensor(d!) valid) -> ()");
  m.def(
      "grouped_projection_resident_out(Tensor descriptors, Tensor(a!)[] owners, int jobs, int m, int n, int k) -> ()");
  m.def(
      "grouped_projection_resident(Tensor[] x, Tensor[] scale, Tensor[] bias, Tensor[] packed_zn, Tensor[] pair_lut) "
      "-> Tensor[]");
  m.def("make_constants(Tensor anchor) -> Tensor");
  // Every writable descriptor target is an owner. Conservatively mark the
  // complete list mutable; this internal out op is not a functional graph op.
  m.def(
      "grouped_projection_out(Tensor descriptors, Tensor constants, Tensor(a!)[] owners, int jobs, int m, "
      "int n, int k, int tiles, bool pipeline=False) -> ()");
  m.def("projection(Tensor x, Tensor scale, Tensor bias, Tensor packed, Tensor book, Tensor ids) -> Tensor");
  m.def("cube_control(Tensor x, Tensor weight, bool bridge=False) -> Tensor");
  m.def(
      "grouped_projection(Tensor[] x, Tensor[] scale, Tensor[] bias, Tensor[] packed, Tensor[] book, Tensor[] ids) -> "
      "Tensor[]");
  m.def(
      "grouped_projection_pipeline(Tensor[] x, Tensor[] scale, Tensor[] bias, Tensor[] packed, Tensor[] book, "
      "Tensor[] ids) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(vq2a8_ascendc_v3, PrivateUse1, m) {
  m.impl("prepare_out", &vq2a8_v3::PrepareOut);
  m.impl("grouped_projection_resident_out", &vq2a8_v3::GroupedProjectionResidentOut);
  m.impl("grouped_projection_resident", &vq2a8_v3::GroupedProjectionResident);
  m.impl("make_constants", &vq2a8_v3::MakeConstants);
  m.impl("grouped_projection_out", &vq2a8_v3::GroupedProjectionOut);
  m.impl("projection", &vq2a8_v3::Projection);
  m.impl("cube_control", &vq2a8_v3::CubeControl);
  m.impl("grouped_projection", &vq2a8_v3::GroupedProjection<false>);
  m.impl("grouped_projection_pipeline", &vq2a8_v3::GroupedProjection<true>);
}
