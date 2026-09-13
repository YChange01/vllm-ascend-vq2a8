# VQ2A8 TP2 离线 packed-zN 生成

`tools/repack_vq2a8_tp2.py` 从**最原始 canonical `experts_vq`** 直接生成两份 TP rank
权重，提前完成 permutation 吸收、TP2 分片、K256 补位和 zN/FP8 pair-LUT 转换。
原始文件不修改，不需要先生成 TP1 文件，不展开 dense BF16 权重，不使用 NPU。

**边界：这是新的离线权重格式，不是已经接通的 TP2 服务。**
当前 V3 loader、native shape checks 和 serving 仍是 TP1 合同；不能把输出传给现有
TP1 入口，也不能仅加 `--tensor-parallel-size 2` 就认为接入完成。
旧版、V2、V3 的默认运行路径均未改变。

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
两个 rank 都包含相同 expert IDs，未来运行时必须路由到相同 token/expert。

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
`packed_shape` 可能不同；未来 loader 必须读每个矩阵的 metadata，不能假设所有 K 一致。

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

## 数值与后续模型接入

CPU 回归独立检查 canonical 权重映射、gate/up 配对、任意跨 rank permutation、dummy、
局部 RHT/bias 及序列化；但没有在 NPU 上执行模型。

该合同采用 GPU 参考式 **TP2-local A8**：每个 rank 按自己 K 的 amax 量化。它与 TP1
全 K 的 amax 不同，再加 K regrouping 和跨 rank SUM 的归约顺序变化，不能承诺 TP1
逐位一致。未来必须做数值与模型质量验证；若改成跨 rank MAX 来共享量化 scale，应另行
明确 runtime/格式合同，而不是静默改变当前描述。

后续至少需要 TP2 artifact loader、native 的 N2048/真实 K 与 padded K 支持、局部激活
准备、TP collective、普通权重 TP/sharing 策略，以及完整模型数值/峰值内存/性能实测。
本脚本把昂贵的权重布局转换移到离线，但**不会自动接通上述运行路径**。
