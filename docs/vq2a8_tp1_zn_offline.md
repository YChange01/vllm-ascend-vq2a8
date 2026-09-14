# VQ2A8 TP1 离线 packed-zN

本分支基于 vLLM Ascend v0.23.0，环境安装见 [v0.23 迁移说明](vq2a8_v023_migration.md)。
已有 TP1 ZN 文件可复用；迁移框架本身不要求重新 repack。

这是与 TP2 相同的 packed-zN / K256 pair-LUT 布局，不是旧
`repack_vq2a8_tp1.py` 生成的 `vq2a8_direct_tp1_v1`。新格式为
`vq2a8_zn_tp1_v1`，仅包含 `tp1/rank0`，保留完整 gate/up 和 down。
旧 TP1 direct、旧版本、V2 以及 TP2 的格式和默认入口保持兼容。

## 从原始权重生成

输入必须是原始 canonical `experts_vq`，不是 TP2 rank 目录，也不是旧 TP1 direct 目录。
输出必须是全新的目录；不改源文件、不覆盖已生成的 TP1/TP2，不重新量化、不展开 dense BF16。
仅需 CPU PyTorch、NumPy、safetensors，不导入 vLLM/torch_npu，不占用 NPU。

先估算文件大小，不生成文件：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/repack_vq2a8_tp1_zn.py \
  --input /home/g00872988/vq2a8/experts_vq \
  --output /home/g00872988/vq2a8/experts_vq_tp1_zn \
  --model-config /home/g00872988/vq2a8/config.json \
  --dry-run
```

完整生成：

```bash
python -u tools/repack_vq2a8_tp1_zn.py \
  --input /home/g00872988/vq2a8/experts_vq \
  --output /home/g00872988/vq2a8/experts_vq_tp1_zn \
  --model-config /home/g00872988/vq2a8/config.json \
  --layers all \
  --threads 4 \
  --experts-per-shard 32
```

如果原始目录名实际为 `expert_vq`，只改 `--input`；内部仍须是
`experts_vq_layer_N.json/.safetensors` 配对文件。输出存在时换一个新目录名，不要先删除旧产物。
`--experts-per-shard 8` 可降低临时主机内存；`--layers 3` 可生成单层试件，
但它会标记 `complete=false`，不能拿来启动完整服务。

终端打印格式检查、源哈希、expert 转换、分片写入回读校验和完成阶段。
每层只打开一次源 safetensors，分批保存；发布前复核全部选中源文件、配置和转换代码的哈希。
发布使用不覆盖目标的原子操作；失败只清理本次私有 staging，不把半成品当作成品。

## 磁盘与算子布局

| 字段 | dtype / 每 expert 的形状 | 含义 |
| --- | --- | --- |
| `packed_zn` | uint8 `[N/32,K/16,16,8]` | 沿 N 打包相邻两个 4-bit VQ pair 索引 |
| `pair_lut` | uint8 `[K/256,N/32,32]` | 原始 FP8 E4M3FN 向量码本字节，不是 INT4/FP4 标量权重 |
| `activation_order` | int64 `[K]` | packed K 到原物理 K 的稳定 gather 映射 |
| `weight_scale` / `weight_bias` | float32 `[K]` | 保持原物理 K 顺序，不预先按码本重排 |
| `rht_sign` | int8 `[K]` | 原物理 K 上的 RHT128 符号 |

与 TP2 的区别只在分片和通信语义，不改变 zN 字节布局：

| 投影 | 本模型 TP1 完整 N×K | 本模型 TP2 每 rank |
| --- | --- | --- |
| Gate/up | `4096×4096` | `2048×4096`，各 rank 保留成对 gate/up 通道 |
| Down | `4096×2048` | 真实 `4096×1024`，按码本补列后由 runtime 对齐 K2048 |

TP1 每个源码本恰好保留 256 列，因此 `logical_shape == packed_shape`、`padding_columns=0`。
TP1 gate/up 保持 `[完整 gate, 完整 up]`，down 保留完整输入 K；没有 TP SUM/HCCL。
激活依次做物理 RHT128、bias GEMV/weight scale、完整 K 上的动态 FP8 量化，
最后按 `activation_order` 搬运 FP8 字节。不能把 gather 移到 RHT/量化之前。

## V3 直接加载

新 packed-zN 仅接入 V3 resident/V2 projection 路径；不会回退给旧 direct 或 legacy kernel。
loader 严格检查格式、完整层/expert、源配置、SHA、形状和映射；每个分片读取一次，
把已转换的六字段复制到最终设备 bank，不再执行 `convert_expert_payload`。
旧 direct artifact 仍沿用原来的启动转换路径。

先确认所选卡空闲，再启动（示例卡 0，不表示卡 0 当前可用）：

```bash
python -u tools/serve_vq2a8_v3.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_tp1_zn \
  --tensor-parallel-size 1 \
  --physical-npu 0 \
  --preparation eager \
  --decode-graph none
```

没有修改 native C++、设备 ABI 或计算分块：本次 Python 更新不要求重新编译已有兼容 V3 库。
加载仍有磁盘读取、校验和 H2D 时间；离线 repack 不减少最终常驻权重 HBM，
原有预算与预留检查必须继续通过，不保证单卡 86 GB 能容纳完整服务。

## 验证边界

CPU 回归逐字节比较新 TP1 六字段与旧
`convert_expert_payload(repack_matrix_tp1(canonical, spec), spec)` 的结果，
覆盖本模型两种真实 N/K、恒等/随机/逆序/跨 K256 置换、FP8 正负零/边界字节，
并独立解码核对完整物理权重。另验证分片序列化、篡改拒绝和直载不调用转换。

这些检查不是 NPU 数值、模型质量、显存峰值或 TPOT 验证。manifest 中保守的
`runtime_compatible=false` / `runtime_supported=false` 保留设备未验证含义，
不应手工改成 true。与旧 V3 启动转换的字节一致也不等于与 V1 推理逐 bit 一致。
