// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "kernel_operator.h"
#include "launch.h"
#define VQ2A8_LAYOUT_FN __aicore__ inline
#include "layout.h"

// CANN's 1:1 TSCM compatibility mode can implement UB->L1 through GM.
// Reject that route instead of silently changing the on-chip contract.
#if defined(__MIX_CORE_AIC_RATION__) && __MIX_CORE_AIC_RATION__ == 1
  #error "VQ2A8 requires native A5 1C:2V UB-to-L1, not TSCM GM compatibility"
#endif

// A5-only, single-buffer correctness prototype. Decode uses scalar accesses
// to UB, not scalar GM gathers. Vectorized lookup/pipelining are future work.
// API/layout precedents: attention/kv_quant_sparse_attn_sharedkv/op_kernel/arch35
// common/{matmul,buffer,FixpipeOut}.h. No Triton, MX scales or dense workspace.
namespace vq2a8_ascendc {
using namespace AscendC;
constexpr uint32_t kReady = 0, kRead = 1, kResult = 2, kStored = 3;
constexpr uint32_t kPeer = 16;  // A5 mode 4: AIV1 flags appear at id+16 on AIC
constexpr FixpipeConfig kToUb = {CO2Layout::ROW_MAJOR, true};

template <HardEvent E>
__aicore__ inline void Fence() {
  // TPipe owns/pre-sets some events (A5 M_MTE1 IDs 0,1,2). Do not
  // overwrite those tokens or consume the flags reserved for its teardown.
  event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
  SetFlag<E>(event);
  WaitFlag<E>(event);
}

class ProjectionKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR packed, GM_ADDR book, GM_ADDR ids,
                              GM_ADDR dense, GM_ADDR y, uint32_t m, uint32_t n, uint32_t k, uint32_t tiles) {
    m_ = m;
    n_ = n;
    k_ = k;
    tiles_ = tiles;
    x_.SetGlobalBuffer(x);
    if (book != nullptr) {
      book_.SetGlobalBuffer(book);
    }
    if (ids != nullptr) {
      ids_.SetGlobalBuffer(ids);
    }
    if (dense != nullptr) {
      dense_.SetGlobalBuffer(dense);
    }
    if (packed != nullptr) {
      packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(packed));
    }
    scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
    y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    // Identical buffer allocation order on AIC/AIV: Fixpipe addresses the
    // result UB at offset zero on BOTH AIVs. L1 is shared by the core group.
    pipe_.InitBuffer(result_, kHalf * kN * sizeof(float));
    pipe_.InitBuffer(aL1_, kM * kK);
    pipe_.InitBuffer(bL1_, kN * kK);
    pipe_.InitBuffer(aL0_, kM * kK);
    pipe_.InitBuffer(bL0_, kN * kK);
    pipe_.InitBuffer(cL0_, kM * kN * sizeof(float));
    pipe_.InitBuffer(nd_, kHalfTileBytes);
    pipe_.InitBuffer(aUb_, kHalfTileBytes);
    pipe_.InitBuffer(bUb_, kHalfTileBytes);
    pipe_.InitBuffer(bookUb_, kMaxTiles * kN);
    pipe_.InitBuffer(packedUb_, 8 * (kK / 8) * sizeof(uint32_t));
    pipe_.InitBuffer(idsUb_, kK);
    pipe_.InitBuffer(scaleUb_, kHalf * sizeof(float));
    pipe_.InitBuffer(biasUb_, kHalf * sizeof(float));
    pipe_.InitBuffer(outUb_, kHalf * kN * sizeof(bfloat16_t));
  }

  template <uint32_t Mode>
  __aicore__ inline void Process(uint32_t cores) {
    uint32_t core;
    if ASCEND_IS_AIC {
      core = GetBlockIdx();
    } else {
      core = GetBlockIdx() / 2;
    }
    // Host passes the AIC count explicitly, avoiding core-dependent
    // interpretation of GetBlockNum() in MIX kernels.
    for (uint32_t group = core; group < n_ / kN; group += cores) {
      if ASCEND_IS_AIC {
        Cube();
      } else {
        Vector<Mode>(group);
      }
    }
  }

 private:
  // Transfer/pad a 16-row ND tile, then reorder WORDS in UB. All source
  // transfers cover real rows only; M=1 does not read a padded GM row.
  __aicore__ inline void LoadNd(GlobalTensor<uint8_t>& src, uint64_t offset, uint32_t rows, LocalTensor<uint8_t> dst,
                                bool flip) {
    auto ndTile = nd_.Get<uint8_t>();
    auto ndWords = ndTile.ReinterpretCast<int32_t>();
    Duplicate(ndWords, int32_t(0), kHalfTileBytes / 4);
    Fence<HardEvent::V_MTE2>();
    if (rows != 0) {
      DataCopyParams p;
      p.blockCount = rows;
      p.blockLen = kK / 32;
      p.srcStride = (k_ - kK) / 32;
      p.dstStride = 0;
      DataCopy(ndTile, src[offset], p);
    }
    Fence<HardEvent::MTE2_S>();
    auto source = ndTile.ReinterpretCast<uint32_t>();
    auto target = dst.ReinterpretCast<uint32_t>();
    for (uint32_t row = 0; row < kHalf; ++row) {
      for (uint32_t col = 0; col < kK; col += 4) {
        uint32_t value = source.GetValue((row * kK + col) / 4);
        target.SetValue(HalfNz(row, col) / 4, value ^ (flip ? 0x80808080u : 0u));
      }
    }
    // Scalar reads of ND must finish before the next Vector zero-fill.
    Fence<HardEvent::S_V>();
  }

  __aicore__ inline void Decode(uint32_t group, uint32_t start, uint32_t half) {
    auto words = packedUb_.Get<uint32_t>();
    auto ids = idsUb_.Get<uint8_t>();
    DataCopyParams p;
    p.blockCount = 8;
    p.blockLen = kK / 64;
    p.srcStride = (k_ - kK) / 64;
    p.dstStride = 0;
    DataCopy(words, packed_[PackedOffset(group * kN + half * kHalf, start, k_)], p);
    DataCopy(ids, ids_[start], kK);
    Fence<HardEvent::MTE2_S>();
    auto table = bookUb_.Get<uint16_t>();
    auto dst = bUb_.Get<uint32_t>();
    auto tileWords = ids.ReinterpretCast<uint32_t>();
    // Reuse each packed word for both rows and all eight columns. Lookup
    // both FP8 bytes together and write four adjacent NZ bytes at a time.
    // Still scalar UB decode, but no longer a GetValue/SetValue per byte.
    for (uint32_t row = 0; row < kHalf; row += 2) {
      for (uint32_t col = 0; col < kK; col += 8) {
        uint32_t codes = words.GetValue(PackedOffset(row, col, kK));
        for (uint32_t offset = 0; offset < 8; offset += 4) {
          uint32_t even, odd;
          DecodeFour(codes >> (offset * 4), tileWords.GetValue((col + offset) / 4), tiles_, table, even, odd);
          dst.SetValue(HalfNz(row, col + offset) / 4, even);
          dst.SetValue(HalfNz(row + 1, col + offset) / 4, odd);
        }
      }
    }
    Fence<HardEvent::S_MTE2>();  // packed/IDs can be overwritten next iteration
  }

  __aicore__ inline void CopyHalfToL1(LocalTensor<uint8_t> dst, LocalTensor<uint8_t> src, uint32_t half) {
    DataCopyParams p;
    p.blockCount = kK / kC0;
    p.blockLen = kHalf;
    p.srcStride = 0;
    p.dstStride = kHalf;
    DataCopy(dst[half * kHalf * kC0], src, p);
  }

  template <uint32_t Mode>
  __aicore__ inline void Vector(uint32_t group) {
    uint32_t half = GetSubBlockIdx();
    uint32_t firstRow = half * kHalf;
    uint32_t rows = HalfRows(m_, firstRow);
    if constexpr (Mode == 2) {
      DataCopyParams p;
      p.blockCount = tiles_;
      p.blockLen = 1;
      p.srcStride = n_ / 32 - 1;
      p.dstStride = 0;
      DataCopy(bookUb_.Get<uint8_t>(), book_[group * kN], p);
      Fence<HardEvent::MTE2_S>();
    }
    if (rows != 0) {
      DataCopyExtParams p{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
      DataCopyPadExtParams<float> pad{false, 0, 0, 0};
      DataCopyPad(scaleUb_.Get<float>(), scale_[firstRow], p, pad);
      DataCopyPad(biasUb_.Get<float>(), bias_[firstRow], p, pad);
      Fence<HardEvent::MTE2_S>();
    }
    for (uint32_t start = 0; start < k_; start += kK) {
      LoadNd(x_, uint64_t(firstRow) * k_ + start, rows, aUb_.Get<uint8_t>(), false);
      if constexpr (Mode == 2) {
        Decode(group, start, half);
      } else {
        LoadNd(dense_, uint64_t(group * kN + half * kHalf) * k_ + start, kHalf, bUb_.Get<uint8_t>(), Mode == 1);
      }
      Fence<HardEvent::S_MTE3>();
      CopyHalfToL1(aL1_.Get<uint8_t>(), aUb_.Get<uint8_t>(), half);
      CopyHalfToL1(bL1_.Get<uint8_t>(), bUb_.Get<uint8_t>(), half);
      CrossCoreSetFlag<4, PIPE_MTE3>(kReady);
      CrossCoreWaitFlag<4, PIPE_MTE3>(kRead);
      // Drain both the L1 handoff and UB reads before scalar reuse.
      Fence<HardEvent::MTE3_S>();
    }
    CrossCoreWaitFlag<4, PIPE_V>(kResult);
    auto result = result_.Get<float>();
    auto out = outUb_.Get<bfloat16_t>();
    for (uint32_t row = 0; row < rows; ++row) {
      Muls(result[row * kN], result[row * kN], scaleUb_.Get<float>().GetValue(row), kN);
      PipeBarrier<PIPE_V>();
      Adds(result[row * kN], result[row * kN], biasUb_.Get<float>().GetValue(row), kN);
      PipeBarrier<PIPE_V>();
    }
    Cast(out, result, RoundMode::CAST_RINT, kHalf * kN);
    Fence<HardEvent::V_MTE3>();
    if (rows != 0) {
      DataCopyParams p;
      p.blockCount = rows;
      p.blockLen = kN * sizeof(bfloat16_t) / 32;
      p.srcStride = 0;
      p.dstStride = (n_ - kN) * sizeof(bfloat16_t) / 32;
      DataCopy(y_[uint64_t(firstRow) * n_ + group * kN], out, p);
    }
    CrossCoreSetFlag<4, PIPE_MTE3>(kStored);
    Fence<HardEvent::MTE3_S>();
  }

  __aicore__ inline void Cube() {
    auto a = aL0_.Get<fp8_e4m3fn_t>();
    auto b = bL0_.Get<fp8_e4m3fn_t>();
    auto c = cL0_.Get<float>();
    LoadData2DParamsV2 load;
    load.mStartPosition = 0;
    load.kStartPosition = 0;
    load.mStep = 2;
    load.kStep = 4;
    load.srcStride = 2;
    load.dstStride = 2;
    load.ifTranspose = false;
    for (uint32_t start = 0; start < k_; start += kK) {
      CrossCoreWaitFlag<4, PIPE_MTE1>(kReady);
      CrossCoreWaitFlag<4, PIPE_MTE1>(kReady + kPeer);
      LoadData(a, aL1_.Get<fp8_e4m3fn_t>(), load);
      LoadData(b, bL1_.Get<fp8_e4m3fn_t>(), load);  // B stored as [N,K]
      CrossCoreSetFlag<4, PIPE_MTE1>(kRead);
      CrossCoreSetFlag<4, PIPE_MTE1>(kRead + kPeer);
      Fence<HardEvent::MTE1_M>();
      MmadParams p;
      p.m = kM;
      p.n = kN;
      p.k = kK;
      p.cmatrixInitVal = start == 0;
      p.cmatrixSource = false;
      p.unitFlag = 0;
      Mmad(c, a, b, p);  // native E4M3 x E4M3 -> FP32; no MX/FP16 fallback
      Fence<HardEvent::M_MTE1>();
    }
    Fence<HardEvent::M_FIX>();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> p;
    p.nSize = kN;
    p.mSize = kM;
    p.srcStride = kM;
    p.dstStride = kN;
    p.dualDstCtl = 1;  // first 16 rows to AIV0, last 16 to AIV1
    p.params.ndNum = 1;
    p.params.srcNdStride = 0;
    p.params.dstNdStride = 0;
    Fixpipe<float, float, kToUb>(result_.Get<float>(), c, p);
    CrossCoreSetFlag<4, PIPE_FIX>(kResult);
    CrossCoreSetFlag<4, PIPE_FIX>(kResult + kPeer);
    CrossCoreWaitFlag<4, PIPE_FIX>(kStored);
    CrossCoreWaitFlag<4, PIPE_FIX>(kStored + kPeer);
    Fence<HardEvent::FIX_M>();
  }

  TPipe pipe_;
  TBuf<TPosition::A1> aL1_, bL1_;
  TBuf<TPosition::A2> aL0_;
  TBuf<TPosition::B2> bL0_;
  TBuf<TPosition::CO1> cL0_;
  TBuf<TPosition::VECIN> result_;
  TBuf<TPosition::VECCALC> nd_, aUb_, bUb_, bookUb_, packedUb_, idsUb_, scaleUb_, biasUb_, outUb_;
  GlobalTensor<uint8_t> x_, book_, ids_, dense_;
  GlobalTensor<uint32_t> packed_;
  GlobalTensor<float> scale_, bias_;
  GlobalTensor<bfloat16_t> y_;
  uint32_t m_, n_, k_, tiles_;
};
}  // namespace vq2a8_ascendc

#define VQ2_KERNEL(NAME, MODE)                                                                                     \
  extern "C" __global__ __aicore__ void NAME(GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR packed, GM_ADDR book, \
                                             GM_ADDR ids, GM_ADDR dense, GM_ADDR y, uint32_t m, uint32_t n,        \
                                             uint32_t k, uint32_t tiles, uint32_t cores) {                         \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                                             \
    vq2a8_ascendc::ProjectionKernel op;                                                                            \
    op.Init(x, scale, bias, packed, book, ids, dense, y, m, n, k, tiles);                                          \
    op.Process<MODE>(cores);                                                                                       \
  }
VQ2_KERNEL(vq2a8_ascendc_direct, 0)
VQ2_KERNEL(vq2a8_ascendc_bridge, 1)
VQ2_KERNEL(vq2a8_ascendc_fused, 2)
#undef VQ2_KERNEL

namespace vq2a8_ascendc {
void Launch(void* stream, uint32_t blocks, void* x, void* scale, void* bias, void* packed, void* book, void* ids,
            void* dense, void* y, uint32_t m, uint32_t n, uint32_t k, uint32_t tiles, uint32_t mode) {
#define VQ2_ARGS                                                                                                      \
  static_cast<GM_ADDR>(x), static_cast<GM_ADDR>(scale), static_cast<GM_ADDR>(bias), static_cast<GM_ADDR>(packed),     \
      static_cast<GM_ADDR>(book), static_cast<GM_ADDR>(ids), static_cast<GM_ADDR>(dense), static_cast<GM_ADDR>(y), m, \
      n, k, tiles, blocks
  if (mode == 0) {
    vq2a8_ascendc_direct<<<blocks, nullptr, stream>>>(VQ2_ARGS);
  } else if (mode == 1) {
    vq2a8_ascendc_bridge<<<blocks, nullptr, stream>>>(VQ2_ARGS);
  } else {
    vq2a8_ascendc_fused<<<blocks, nullptr, stream>>>(VQ2_ARGS);
  }
#undef VQ2_ARGS
}
}  // namespace vq2a8_ascendc
