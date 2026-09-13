# VQ2A8 TP2 离线 packed-zN 生成

`tools/repack_vq2a8_tp2.py` 从**最原始 canonical `experts_vq`** 直接生成两份 TP rank
权重，提前完成 permutation 吸收、TP2 分片、K256 补位和 zN/FP8 pair-LUT 转换。
原始文件不修改，不需要先生成 TP1 文件，不展开 dense BF16 权重，不使用 NPU。

显式 TP2 loader、V2-compatible native shape checks 和 V3 双卡服务入口已接入。
必须使用**完整 TP2 artifact、重新编译的 V3 库，以及 TP2 启动参数**；不能把 TP2
文件传给 TP1 loader。旧版、V2、V3 TP1 默认运行路径均保留。
当前验证是 CPU 合同/算术检查；真实 Ascend950 双卡执行、模型质量和 TPOT 仍须实测。

## 运行

在已安装 CPU 可用的 `torch`（需要 `float8_e4m3fn`）、`numpy`、`safetensors` 的环境运行。
无需安装 vLLM/torch_npu，无需编译 CANN 算子。服务器当前 Python 环境通常已经包含这些包。

输入目录必须包含成对的文件：

```text
/home/g00872988/vq2a8/
  config.json
  experts_vq/
    experts_vq_layer_0.json
    experts_vq_layer_0.safetensors
    ...
```

先只检查格式和估算大小（不生成文件，也不校验全部 tensor 值）：

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/repack_vq2a8_tp2.py \
  --input /home/g00872988/vq2a8/experts_vq \
  --output /home/g00872988/vq2a8/experts_vq_tp2_zn \
  --dry-run
```

正式生成两个 rank：

```bash
python -u tools/repack_vq2a8_tp2.py \
  --input /home/g00872988/vq2a8/experts_vq \
  --output /home/g00872988/vq2a8/experts_vq_tp2_zn \
  --threads 4 \
  --experts-per-shard 32
```

若原始目录实际叫 `expert_vq`，修改 `--input`，但内部文件仍须符合上述 canonical 合同。
`--model-config /path/to/config.json` 可以单独指定模型配置。

想先试一个完整 routed layer、估算本机耗时：

```bash
python -u tools/repack_vq2a8_tp2.py \
  --input /home/g00872988/vq2a8/experts_vq \
  --output /home/g00872988/vq2a8/experts_vq_tp2_zn_layer3_smoke \
  --layers 3 --threads 4 --experts-per-shard 8
```

单层输出会标记 `complete=false`，不能冒充完整模型。全量生成请使用另一个新输出目录。
原始模型的前 3 层仅各存一个导出 expert，不能用第 0 层耗时直接乘 43 来估全量耗时。
其余 40 层各 256 experts，可用第 3 层耗时乘约 40，再考虑存储吞吐波动。

## 进度、内存和安全

- 终端打印 `inspect`、`source_hash`、每 8 个 expert 的 `convert`、每个 rank 的
  `write_verify`、`source_recheck`、`layer_done` 和最终 `done`。
- 每层只打开一次 canonical safetensors，按小批 expert 写出；默认 32 个，不把 43 层
  输出同时留在 RAM。降低 `--experts-per-shard` 可以进一步降低 RAM。
- 同一批两份 rank 输出会暂时驻留 CPU，外加单矩阵 uint4 解包网格、校验和序列化临时量；
  它不是零拷贝或常量几 MB 内存，也不是 NPU 显存使用量。
- 发布前检查源文件转换前后 SHA256、模型配置/生成代码 SHA256；每个输出文件写后重新
  读取校验 tensor dtype/shape/value，并记录文件及 metadata SHA256。
- 全部成功才发布新目录。拒绝覆盖已有输出、源/输出互相包含，以及 canonical 文件符号链接。
- 支持 Linux/Windows 原子发布；Linux 使用 `renameat2(RENAME_NOREPLACE)`。开始转换前
  会用私有空目录探测文件系统支持，缺少该能力就提前失败，不降级成可覆盖目标的 rename。
- 首版不提供断点恢复；普通异常/中断会清理本次私有 staging。强制杀进程或断电可能留下
  `.OUTPUT.partial-*`，不能把它当作完成文件；确认已无转换进程后才考虑清理对应临时目录。
- 不会复制 attention、embedding、shared expert、tokenizer 等普通模型文件。

## TP2 的切分含义

这是真正的权重 tensor parallel 分片，不是把 256 个 MoE experts 分成每卡 128 个。
两个 rank 都包含相同 expert IDs，运行时使用复制的 router/hash 根权重及相同输入，
路由到相同 token/expert；这不是 expert parallel。

| 矩阵 | TP2 逻辑分片 | 运行时要求 |
| --- | --- | --- |
| Gate/up `[2I,H]` | 分别取 gate 的 `I/2` 行和 up 的相同 `I/2` 行，再拼成 `[gate_rank,up_rank]` | 同一完整 H 激活输入；本 rank 做对应 SwiGLU |
| Down `[H,I]` | 吸收 canonical permutation 后，切连续物理输入列 `I/2`，输出仍是完整 H | 两 rank 的 down partial 做 SUM；局部 bias correction 只加入本地贡献 |

不能直接把融合 gate/up 矩阵上下对半分，否则 rank 0 只得到 gate、rank 1 只得到 up。
Down 切点必须对齐 RHT128，且必须在物理列坐标切分，不能把排序后的 K 随意砍半。

### 为什么 down 还需要补列

原始每个 codebook 对应 K256，但 permutation 会把其列分散到两个 TP rank。
例如本地 canonical 第 3 层 expert 0 的 down，rank 0 各码本分别留下
`[154,157,138,111,133,122,101,108]` 列，rank 1 是
`[102,99,118,145,123,134,155,148]`。直接聚集以后也不能组成每块仅一个 LUT 的 K256。

新格式为每个非空码本补零激活列到 256：

1. 真实 local K 在前，dummy physical 列追加在后，追加的总长度对齐 RHT128。
2. dummy 激活必须为零，metadata 为 `scale=0,bias=0,sign=+1`。
3. 完成物理 RHT/normalization/FP8 quantization 后，以 `activation_order` gather。
4. dummy code 可以为合法 code 0，无需假设码本恰好存在零权重，因为相乘的激活是零。

没有裁剪真实列、重新拟合码本或改写 FP8 字节。空码本不占 K256，因此不同 expert 的
`packed_shape` 可能不同。新 loader 逐矩阵校验 metadata，并在装入最终设备 bank 时，
将 down 统一补到计算 K2048，不能直接按磁盘 K 发射内核。

对当前 `H=4096,I=2048` 的模型：

- Gate/up：每 rank `[2048,4096]`，packed 权重约 2 MiB。
- Down：真实每 rank `[4096,1024]`，混合 permutation 下补到 `[4096,2048]`，约 2 MiB。
- 原 TP1 两矩阵约 `4+2=6 MiB`，新每 rank 约 `2+2=4 MiB`，因此不是减半，而是约减少三分之一。
- 本地完整 43 层 canonical header 实测预估：每 rank payload 上界 **41.6333 GiB**，
  双 rank 共 **83.2666 GiB**；建议新输出磁盘留至少 100 GiB。
- 这些数字不含普通模型权重、KV cache、workspace、图内存、allocator 和其他进程占用。
  不能据此声明 86 GB 卡一定装得下完整服务或保证 20 ms TPOT。

## 输出合同

```text
experts_vq_tp2_zn/
  manifest.json
  tp2/
    rank0/layer_003/experts_0000_0032.safetensors
    rank0/layer_003/experts_0000_0032.json
    ...
    rank1/layer_003/experts_0000_0032.safetensors
    rank1/layer_003/experts_0000_0032.json
    ...
```

格式标识为 `vq2a8_zn_tp2_v1`。分片范围末端 exclusive；tensor key 是
`EXPERT_ID.gate_up.FIELD` 或 `EXPERT_ID.down.FIELD`。为支持每个 expert 的不同 padded K，
不对所有 expert 张量强行 stack。

| Field | dtype / shape | 含义 |
| --- | --- | --- |
| `packed_zn` | uint8 `[Nlocal/32,Kpadded/16,16,8]` | 同一 K 处沿 N 打包两个 4-bit pair code（四个 scalar 权重） |
| `pair_lut` | uint8 `[Kpadded/256,Nlocal/32,32]` | 原始 FP8 E4M3FN 字节，不是 uint8 数值权重 |
| `activation_order` | int64 `[Kpadded]` | packed K → 扩展的本 rank 物理 K 索引，是完整置换 |
| `weight_scale`, `weight_bias` | float32 `[Kpadded]` | 本地物理列 metadata，dummy 为零 |
| `rht_sign` | int8 `[Kpadded]` | 本地物理列 RHT sign，dummy 为 +1 |

矩阵 metadata 明确 `canonical_shape`、`logical_shape`、`packed_shape`、输入/输出范围、
各 LUT 的源码本编号和有效列数。根 manifest 的 `complete` **只表示层是否齐全**。
`runtime_compatible=false` / `runtime_supported=false` 是刻意保留的状态，不应手工改为 true。

## V2 算子形状核对

V3 resident projection 沿用 V2 的 N128 / AIC K1024 / AIV K512 / MMAD K256
及双缓冲流水，计算内核没有为 TP2 重写。

| 投影 | TP2 真实输入 K | 磁盘 packed K | native N × K |
| --- | ---: | --- | --- |
| Gate/up | 4096 | 4096 | 2048 × 4096 |
| Down | 1024 | 1024 / 1280 / 1536 / 1792 / 2048 | 4096 × 2048 |

K256 是码表块，不是计算尾块。当前 V2 流水收尾会等待两个 K1024 槽，K1024
会缺少第二槽的完成信号，非 K1024 尾块也不会完整计算，因此不能简单放开 K 检查。
补位只复制 packed 字节/metadata 并填零，不重新解码或拟合权重。计算 bank 的预算
按统一 K2048 计，不按更小的磁盘 K 计。TP2 gate 的 N2048 恰好是 16 个 N128 块。

新 V3 库必须具有 `RESIDENT_TP2_PROJECTION` capability bit 4；旧库提前拒绝。
原 V2 目录和 V3 resident 计算流水保持不变。

## 编译与双卡服务

选择两张空闲、驱动/CANN 正常的 Ascend950 卡；不要重置其他任务正在使用的卡。
以下命令不自动生成离线 artifact；先完成上面的全量生成。

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/build_vq2a8_ascendc_v3.py \
  --soc Ascend950DT_9574 --jobs 4
```

建议先做不加载模型的双卡 smoke；只读所选 `.so` 的哈希，不需要验收报告：

```bash
VQ2_V3_LIBRARY="$PWD/build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so"
VQ2_V3_SHA=$(sha256sum "$VQ2_V3_LIBRARY" | cut -d ' ' -f 1)
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  tools/validate_vq2a8_tp2_collective.py \
  --library "$VQ2_V3_LIBRARY" --library-sha256 "$VQ2_V3_SHA" --timeout-s 180
```

终端分阶段输出设备映射、HCCL 初始化、routed SUM、两种 native 局部投影及重复执行。
必须两 rank 都完成；默认不写报告，且不把 CPU/Gloo fallback 当作成功。
`--communication-only` 可独立排查 HCCL，不加载 `.so`；smoke 不证明真实模型质量或 TPOT。

服务终端（卡号按实际空闲卡修改）：

```bash
python -u tools/serve_vq2a8_v3.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_tp2_zn \
  --tensor-parallel-size 2 \
  --physical-npus 0,1 \
  --preparation eager --decode-graph none
```

内部使用标准 `vllm serve`、`mp` executor、两 rank TP group。原 `--physical-npu`
仍用于 TP1，不与 TP2 卡列表混用。支持 `--dry-run` 查看实际启动参数。
服务就绪后，另一个终端：

```bash
python tools/benchmark_vq2a8_serving.py --max-tokens 32 --repeats 3
```

计时为客户端 HTTP streaming TTFT/TPOT，不包含加载时间。首轮先用 eager；之后可单独
改成 `--preparation fused` 测局部准备融合，TP2 的 `--decode-graph` 目前只允许 `none`。
本入口仍限 TP2、B1、context ≤128，未开放 EP/PP/DP/CP、整模型图或 MTP。

### 一键排查容器和 HCCL 通信

无需模型、artifact 或编译库，在仓库根目录运行：

```bash
bash tools/vq2_tp2_diagnose.sh
```

脚本固定测试物理卡 0、1：先收集容器设备、UB 库、驱动、验签状态和环境，
再分别启动原生 PyTorch HCCL 和 vLLM communication-only 测试。
每次通信测试前都要求 `npu-smi` 明确显示两卡无进程；占用、查询失败或无法识别时跳过，
环境检查仍继续。空闲检查只是快照，请确保测试期间不会有新任务占用这两张卡。

每项检查有超时，通信测试每 5 秒打印等待状态，完成后打印阶段、首批错误和退出码。
完整输出在新建的 `/tmp/vq2-tp2-diag.*` 目录，末尾 `UPLOAD=...tar.gz` 给出打包路径。
脚本不安装软件、不修改验签或通信配置、不重置设备；中断时只终止本次启动的测试。
缺少诊断工具或可选文件不单独构成通信故障结论；两个短测通过也不代表模型质量或 TPOT 达标。
分享日志前检查其中的设备地址、环境路径等运维信息。

### Loader 与通信边界

- 每 worker 绑定自己的 artifact rank，验证完整覆盖、config/metadata SHA、两个 rank
  的 shape/分片映射及本 rank tensor SHA；按 shard 一次打开读取，不再逐 expert
  执行原始 permutation/zN 转换。启动仍需文件读取、校验、H2D，并非零加载开销。
- Gate/up 分列输出，本地 SwiGLU 后进入 down。每层在 routed top-k 加权和 scaling
  完成后，对 **FP32 routed partial** 做一次 TP SUM，再加复制的 shared 输出，最后 BF16。
  decode 和 bounded prefill 使用相同的归约位置；不对每个 expert 各做通信。
- shared/router/hash 根权重每卡复制；attention/vocab 根权重走标准参数专属 TP loader，
  attn_sink 按本地 heads 分片。attention 的输出投影保留父模型自己的归约，不能在
  decoder 外再重复 SUM。
- 不存在动态换出、TP1/cache fallback 或把不完整 artifact 当成完整模型的降级路径。

## 数值与验证边界

本次 Windows CPU 合同回归：2126 passed、257 skipped、15 个既有失败（2 个 CPU FP8/FMA
平台差异，13 个旧测试的 Linux 路径/PYTHONPATH 假设），未新增失败。TP2/相关 runtime
专项 350 passed，独立双卡 smoke 的 CPU 契约测试 44 passed；这些数字不包含真实 NPU 运行。

CPU 回归独立检查 canonical 权重映射、gate/up 配对、任意跨 rank permutation、dummy、
局部 RHT/bias 及序列化；但没有在 NPU 上执行模型。

该合同采用 GPU 参考式 **TP2-local A8**：每个 rank 按自己 K 的 amax 量化。它与 TP1
全 K 的 amax 不同，再加 K regrouping 和跨 rank SUM 的归约顺序变化，不能承诺 TP1
逐位一致。必须做真实 NPU 数值与模型质量验证；若改成跨 rank MAX 来共享量化 scale，应另行
明确 runtime/格式合同，而不是静默改变当前描述。

已实现的代码接线不等同于实测通过。首次上机应先运行
[TP2 native smoke](../csrc/vq2a8_ascendc_v3/TP2_NATIVE.md)，再验证两卡通信和服务。
完整模型数值、峰值 HBM、长期稳定性、TTFT/TPOT 仍以真实 Ascend950 运行结果为准；
不能由 CPU PASS 推断 20 ms TPOT。
