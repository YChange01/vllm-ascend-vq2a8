// Adapted from the user-supplied internal mat_fp4_aic.cce/mat_fp4_aiv.cce.
// Preserve their register pair-LUT, padded zN UB, K512/K1024 and N128 pipeline.
// VQ2A8 v3 resident kernel, retaining the v2 arithmetic and pipeline.
// Source rights remain unchanged; see ../vq2a8_expert_reference/README.md.
#include "kernel_operator.h"
#include "resident_launch.h"
#define VQ2_V3_RESIDENT_LAYOUT_FN __aicore__ inline
#include "resident_layout.h"
#undef VQ2_V3_RESIDENT_LAYOUT_FN

#if defined(__MIX_CORE_AIC_RATION__) && __MIX_CORE_AIC_RATION__ == 1
  #error "VQ2A8 v3 resident requires native 1C:2V UB-to-L1, not TSCM GM compatibility"
#endif

namespace vq2a8_v3_resident {
using namespace AscendC;
constexpr uint32_t kPeer = 16;
constexpr uint32_t kResultReady = 4;
constexpr uint32_t kResultStored = 5;
constexpr FixpipeConfig kToUb = {CO2Layout::ROW_MAJOR, true};

template <HardEvent E>
__aicore__ inline void Fence() {
  const auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
  SetFlag<E>(event);
  WaitFlag<E>(event);
}

// Allocate events through TPipe, never overwrite its pre-set M_MTE1 tokens.
template <HardEvent E>
class SlotEvents {
 public:
  __aicore__ inline void Allocate() {
    for (uint32_t i = 0; i < kBuffers; ++i) {
      ids_[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<E>());
    }
  }
  __aicore__ inline void Release() {
    for (uint32_t i = 0; i < kBuffers; ++i) GetTPipePtr()->ReleaseEventID<E>(ids_[i]);
  }
  __aicore__ inline void Set(uint32_t slot) { SetFlag<E>(ids_[slot]); }
  __aicore__ inline void Wait(uint32_t slot) { WaitFlag<E>(ids_[slot]); }
  __aicore__ inline void Start() {
    for (uint32_t i = 0; i < kBuffers; ++i) Set(i);
  }
  __aicore__ inline void Drain() {
    for (uint32_t i = 0; i < kBuffers; ++i) Wait(i);
  }

 private:
  event_t ids_[kBuffers];
};

class ResidentProjection {
 public:
  __aicore__ inline void Init() {
    // SAME allocation order on both core kinds: Fixpipe addresses AIV UB offset
    // zero and UB->L1 addresses the AIC's shared L1 buffers.
    pipe_.InitBuffer(result_, kResultBytes);
    pipe_.InitBuffer(aL1_, kBuffers * kMaxM * kAicK);
    pipe_.InitBuffer(bL1_, kBuffers * kN * kAicK);
    pipe_.InitBuffer(aL0_, kL0ABytes);
    pipe_.InitBuffer(bL0_, kL0BBytes);
    pipe_.InitBuffer(cL0_, kL0CBytes);
    pipe_.InitBuffer(packedUb_, kBuffers * kPackedBytes);
    pipe_.InitBuffer(decodedUb_, kBuffers * kDecodedBytes);
    pipe_.InitBuffer(tableUb_, kBuffers * kTableBytes);
    pipe_.InitBuffer(scaleUb_, kRowBytes);
    pipe_.InitBuffer(biasUb_, kRowBytes);
    pipe_.InitBuffer(outUb_, kOutBytes);
    if ASCEND_IS_AIC {
      aFree_.Allocate();
      l0Free_.Allocate();
    } else {
      SetAtomicNone();
      packedFree_.Allocate();
      decodedFree_.Allocate();
      // Full-register loads may include these padded entries; initialize them.
      Duplicate(tableUb_.Get<uint32_t>(), uint32_t(0), kBuffers * kTableBytes / sizeof(uint32_t));
      Fence<HardEvent::V_MTE2>();
    }
  }

  __aicore__ inline void Process(GM_ADDR descriptors, uint32_t jobs, uint32_t nTiles, uint32_t cores) {
    GlobalTensor<uint64_t> records;
    records.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(descriptors));
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
      core /= 2;
    }
    // Dynamic 1C:2V mapping on BOTH sides, including SKUs with fewer than 32 AIC.
    for (uint32_t work = core; work < jobs * nTiles; work += cores) {
      const uint32_t base = work / nTiles * kJobWords;
      x_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t*>(records.GetValue(base + kX)));
      packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(records.GetValue(base + kPacked)));
      table_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(records.GetValue(base + kTable)));
      scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(records.GetValue(base + kScale)));
      bias_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(records.GetValue(base + kBias)));
      output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(records.GetValue(base + kOutput)));
      m_ = records.GetValue(base + kRows);
      n_ = records.GetValue(base + kColumns);
      k_ = records.GetValue(base + kReduction);
      const uint32_t nBegin = (work % nTiles) * kN;
      if ASCEND_IS_AIC {
        Cube();
      } else {
        Vector(nBegin);
      }
    }
    if ASCEND_IS_AIC {
      aFree_.Release();
      l0Free_.Release();
    } else {
      packedFree_.Release();
      decodedFree_.Release();
    }
  }

 private:
  // Restored register types are explicit: uint32 contains two uint16 nibble
  // indices; Gather returns uint16 FP8 pairs, subsequently stored as raw bytes.
  // This accepts arbitrary 16x2 VQ codewords, NOT just four scalar W2 levels.
  __aicore__ inline void RunLut(LocalTensor<uint8_t> packed, LocalTensor<uint8_t> table, LocalTensor<uint8_t> decoded) {
    auto* packedAddr = reinterpret_cast<__ubuf__ uint32_t*>(packed.GetPhyAddr());
    auto* tableAddr = reinterpret_cast<__ubuf__ uint16_t*>(table.GetPhyAddr());
    auto* decodedAddr = reinterpret_cast<__ubuf__ uint8_t*>(decoded.GetPhyAddr());
    if ASCEND_IS_AIV {
      __VEC_SCOPE__ {
        using namespace AscendC::MicroAPI;
        MaskReg all32 = CreateMask<uint32_t, MaskPattern::ALL>();
        MaskReg all8 = CreateMask<uint8_t, MaskPattern::ALL>();
        RegTensor<uint16_t> lut, value;
        RegTensor<uint32_t> word, highNibble, highIndex, index, mask;
        // CANN 9.1 vshr/vshl use signed per-lane counts even for uint32 data.
        RegTensor<int32_t> shiftRight, shiftLeft;
        Duplicate(shiftRight, int32_t(4), all32);
        Duplicate(shiftLeft, int32_t(16), all32);
        Duplicate(mask, uint32_t(0x000F000F), all32);
        for (uint16_t n1 = 0; n1 < kN / kN0; ++n1) {
          for (uint16_t lutK = 0; lutK < kAivK / kCodebookK; ++lutK) {
            DataCopy(lut, tableAddr + (lutK * (kN / kN0) + n1) * 16);
            for (uint16_t localK1 = 0; localK1 < kCodebookK / kK0; ++localK1) {
              const uint16_t k1 = lutK * (kCodebookK / kK0) + localK1;
              const uint32_t packedOffset = (uint32_t(n1) * (kAivK / kK0) + k1) * kK0 * (kN0 / 4);
              const uint32_t decodedOffset = (uint32_t(n1) * (kAivK / kK0) + k1) * (kK0 + 1) * kN0;
              for (uint16_t repeat = 0; repeat < 2; ++repeat) {
                DataCopy<uint32_t, LoadDist::DIST_UNPACK4_B8>(word, packedAddr + (packedOffset + repeat * 64) / 4);
                ShiftRight(highNibble, word, shiftRight, all32);
                ShiftLeft(highIndex, highNibble, shiftLeft, all32);
                Or(index, word, highIndex, all32);
                And(index, index, mask, all32);
                Gather(value, lut, reinterpret_cast<RegTensor<uint16_t>&>(index));
                DataCopy(decodedAddr + decodedOffset + repeat * 256, reinterpret_cast<RegTensor<uint8_t>&>(value),
                         all8);
              }
            }
          }
        }
      }
    }
  }

  __aicore__ inline void LoadVectorTile(uint32_t nBegin, uint32_t kBegin, uint32_t slot) {
    DataCopyParams packedCopy;
    packedCopy.blockCount = kN / kN0;
    packedCopy.blockLen = kAivK * (kN0 / 4) / 32;
    packedCopy.srcStride = (k_ - kAivK) * (kN0 / 4) / 32;
    packedCopy.dstStride = 0;
    DataCopy(packedUb_.Get<uint8_t>()[slot * kPackedBytes], packed_[PackedOffset(nBegin, kBegin, k_)], packedCopy);
    DataCopyParams tableCopy;
    tableCopy.blockCount = kAivK / kCodebookK;
    tableCopy.blockLen = kN / kN0;
    tableCopy.srcStride = (n_ - kN) / kN0;
    tableCopy.dstStride = 0;
    DataCopy(tableUb_.Get<uint8_t>()[slot * kTableBytes], table_[TableOffset(nBegin, kBegin, n_)], tableCopy);
  }

  __aicore__ inline void StoreB1(uint32_t slot, uint32_t half) {
    DataCopyParams copy;
    copy.blockCount = kAivK / kK0;
    copy.blockLen = kK0;
    copy.srcStride = 1;  // Drop the expert's one-row UB bank-conflict padding.
    copy.dstStride = 0;
    for (uint32_t n1 = 0; n1 < kN / kN0; ++n1) {
      DataCopy(bL1_.Get<uint8_t>()[slot * kN * kAicK + B1Offset(n1, half)],
               decodedUb_.Get<uint8_t>()[slot * kDecodedBytes + DecodedOffset(n1, 0)], copy);
    }
  }

  __aicore__ inline void Vector(uint32_t nBegin) {
    const uint32_t half = GetSubBlockIdx();
    const uint32_t halfCapacity = AlignedM(m_) / 2;
    const uint32_t firstRow = half * halfCapacity;
    const uint32_t rows = HalfRows(m_, half);
    if (rows != 0) {
      DataCopyExtParams copy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
      DataCopyPadExtParams<float> pad{false, 0, 0, 0};
      DataCopyPad(scaleUb_.Get<float>(), scale_[firstRow], copy, pad);
      DataCopyPad(biasUb_.Get<float>(), bias_[firstRow], copy, pad);
      Fence<HardEvent::MTE2_S>();
    }
    packedFree_.Start();
    decodedFree_.Start();
    for (uint32_t ki = 0; ki < k_ / kAicK; ++ki) {
      const uint32_t slot = ki % kBuffers;
      packedFree_.Wait(slot);
      LoadVectorTile(nBegin, ki * kAicK + half * kAivK, slot);
      Fence<HardEvent::MTE2_V>();
      decodedFree_.Wait(slot);
      RunLut(packedUb_.Get<uint8_t>()[slot * kPackedBytes], tableUb_.Get<uint8_t>()[slot * kTableBytes],
             decodedUb_.Get<uint8_t>()[slot * kDecodedBytes]);
      packedFree_.Set(slot);
      Fence<HardEvent::V_MTE3>();
      // Decode while the AIC consumes the other K1024 L1 buffer. Reuse is
      // acknowledged after its MTE1 reads, not merely after Mmad was submitted.
      if (ki >= kBuffers) CrossCoreWaitFlag<4, PIPE_MTE3>(2 * slot + 1);
      StoreB1(slot, half);
      decodedFree_.Set(slot);
      CrossCoreSetFlag<4, PIPE_MTE3>(2 * slot);
    }
    for (uint32_t slot = 0; slot < kBuffers; ++slot) CrossCoreWaitFlag<4, PIPE_MTE3>(2 * slot + 1);
    packedFree_.Drain();
    decodedFree_.Drain();
    CrossCoreWaitFlag<4, PIPE_V>(kResultReady);
    auto result = result_.Get<float>();
    auto out = outUb_.Get<bfloat16_t>();
    for (uint32_t row = 0; row < rows; ++row) {
      Muls(result[row * kN], result[row * kN], scaleUb_.Get<float>().GetValue(row), kN);
      PipeBarrier<PIPE_V>();
      Adds(result[row * kN], result[row * kN], biasUb_.Get<float>().GetValue(row), kN);
      PipeBarrier<PIPE_V>();
    }
    // The ONLY BF16 conversion is after original FP32 row scale + bias.
    Cast(out, result, RoundMode::CAST_RINT, halfCapacity * kN);
    Fence<HardEvent::V_MTE3>();
    if (rows != 0) {
      DataCopyParams copy;
      copy.blockCount = rows;
      copy.blockLen = kN * sizeof(bfloat16_t) / 32;
      copy.srcStride = 0;
      copy.dstStride = (n_ - kN) * sizeof(bfloat16_t) / 32;
      DataCopy(output_[uint64_t(firstRow) * n_ + nBegin], out, copy);
    }
    CrossCoreSetFlag<4, PIPE_MTE3>(kResultStored);
    Fence<HardEvent::MTE3_S>();
  }

  __aicore__ inline void LoadA1(uint32_t kBegin, uint32_t slot) {
    // Same GM ND -> L1 zN operation as gm_l1_nd2nz in mat_fp4_aic.cce,
    // expressed by the SDK's typed parameters rather than raw bit fields.
    Nd2NzParams copy;
    copy.ndNum = 1;
    copy.nValue = m_;
    copy.dValue = kAicK;
    copy.srcNdMatrixStride = 0;
    copy.srcDValue = k_;
    copy.dstNzC0Stride = AlignedM(m_);
    copy.dstNzNStride = 1;
    copy.dstNzMatrixStride = 0;
    DataCopy(aL1_.Get<fp8_e4m3fn_t>()[slot * kMaxM * kAicK], x_[kBegin], copy);
  }

  __aicore__ inline void LoadL0(uint32_t aicSlot, uint32_t madOffset, uint32_t madSlot) {
    LoadData2DParamsV2 a;
    a.mStartPosition = 0;
    a.kStartPosition = madOffset / 32;
    a.mStep = AlignedM(m_) / 16;
    a.kStep = kMadK / 32;
    a.srcStride = AlignedM(m_) / 16;
    a.dstStride = AlignedM(m_) / 16;
    a.ifTranspose = false;
    LoadData(aL0_.Get<fp8_e4m3fn_t>()[madSlot * kMaxM * kMadK], aL1_.Get<fp8_e4m3fn_t>()[aicSlot * kMaxM * kAicK], a);
    // Expert B: L1 zN[K0=16,N0=32] -> L0B nZ[K0=32,N0=16].
    // This is NOT the old native kernel's non-transposed N32 load path.
    LoadData2DParamsV2 b;
    b.mStartPosition = madOffset / 16;
    b.kStartPosition = 0;
    b.mStep = kMadK / 16;
    b.kStep = kN / 32;
    b.srcStride = kAicK / 16;
    b.dstStride = kN / 16;
    b.ifTranspose = true;
    LoadData(bL0_.Get<fp8_e4m3fn_t>()[madSlot * kN * kMadK], bL1_.Get<fp8_e4m3fn_t>()[aicSlot * kN * kAicK], b);
  }

  __aicore__ inline void Cube() {
    aFree_.Start();
    l0Free_.Start();
    for (uint32_t ki = 0; ki < k_ / kAicK; ++ki) {
      const uint32_t slot = ki % kBuffers;
      aFree_.Wait(slot);
      LoadA1(ki * kAicK, slot);
      CrossCoreWaitFlag<4, PIPE_MTE1>(2 * slot);
      CrossCoreWaitFlag<4, PIPE_MTE1>(2 * slot + kPeer);
      Fence<HardEvent::MTE2_MTE1>();
      for (uint32_t madOffset = 0; madOffset < kAicK; madOffset += kMadK) {
        const uint32_t madSlot = (madOffset / kMadK) % kBuffers;
        l0Free_.Wait(madSlot);
        LoadL0(slot, madOffset, madSlot);
        Fence<HardEvent::MTE1_M>();
        MmadParams p;
        p.m = AlignedM(m_);
        p.n = kN;
        p.k = kMadK;
        p.disableGemv = true;
        p.cmatrixInitVal = ki == 0 && madOffset == 0;
        p.cmatrixSource = false;
        p.unitFlag = 0;
        // Model preparation already produced E4M3 bytes and a FP32 row scale.
        // Plain FP8 Mmad preserves those bytes: do not requantize to MXFP8 or
        // pretend an arbitrary FP32 scale is representable by an E8M0 exponent.
        Mmad(cL0_.Get<float>(), aL0_.Get<fp8_e4m3fn_t>()[madSlot * kMaxM * kMadK],
             bL0_.Get<fp8_e4m3fn_t>()[madSlot * kN * kMadK], p);
        l0Free_.Set(madSlot);
      }
      CrossCoreSetFlag<4, PIPE_MTE1>(2 * slot + 1);
      CrossCoreSetFlag<4, PIPE_MTE1>(2 * slot + 1 + kPeer);
      aFree_.Set(slot);
    }
    Fence<HardEvent::M_FIX>();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> copy;
    copy.nSize = kN;
    copy.mSize = AlignedM(m_);
    copy.srcStride = AlignedM(m_);
    copy.dstStride = kN;
    copy.quantPre = QuantMode_t::NoQuant;
    copy.dualDstCtl = 1;
    copy.params.ndNum = 1;
    copy.params.srcNdStride = 0;
    copy.params.dstNdStride = 0;
    // SDK resolves the FP32 mode. Deliberately no guessed raw 16<<34/2<<34.
    Fixpipe<float, float, kToUb>(result_.Get<float>(), cL0_.Get<float>(), copy);
    CrossCoreSetFlag<4, PIPE_FIX>(kResultReady);
    CrossCoreSetFlag<4, PIPE_FIX>(kResultReady + kPeer);
    CrossCoreWaitFlag<4, PIPE_FIX>(kResultStored);
    CrossCoreWaitFlag<4, PIPE_FIX>(kResultStored + kPeer);
    Fence<HardEvent::FIX_M>();
    aFree_.Drain();
    l0Free_.Drain();
  }

  TPipe pipe_;
  TBuf<TPosition::VECIN> result_;
  TBuf<TPosition::A1> aL1_, bL1_;
  TBuf<TPosition::A2> aL0_;
  TBuf<TPosition::B2> bL0_;
  TBuf<TPosition::CO1> cL0_;
  TBuf<TPosition::VECCALC> packedUb_, decodedUb_, tableUb_, scaleUb_, biasUb_, outUb_;
  GlobalTensor<fp8_e4m3fn_t> x_;
  GlobalTensor<uint8_t> packed_, table_;
  GlobalTensor<float> scale_, bias_;
  GlobalTensor<bfloat16_t> output_;
  SlotEvents<HardEvent::MTE1_MTE2> aFree_;
  SlotEvents<HardEvent::M_MTE1> l0Free_;
  SlotEvents<HardEvent::V_MTE2> packedFree_;
  SlotEvents<HardEvent::MTE3_V> decodedFree_;
  uint32_t m_, n_, k_;
};
}  // namespace vq2a8_v3_resident

extern "C" __global__ __aicore__ void vq2a8_v3_resident_grouped(GM_ADDR descriptors, uint32_t jobs, uint32_t nTiles,
                                                                uint32_t cores) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  vq2a8_v3_resident::ResidentProjection kernel;
  kernel.Init();
  kernel.Process(descriptors, jobs, nTiles, cores);
}

namespace vq2a8_v3_resident {
void LaunchGrouped(void* stream, uint32_t blocks, void* descriptors, uint32_t jobs, uint32_t nTiles) {
  vq2a8_v3_resident_grouped<<<blocks, nullptr, stream>>>(static_cast<GM_ADDR>(descriptors), jobs, nTiles, blocks);
}
}  // namespace vq2a8_v3_resident
