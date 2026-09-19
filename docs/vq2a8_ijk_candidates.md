# VQ2A8 I/J/K：复用低比特优化策略，独立验收

这些是实验候选，不是已证明提速的默认配置。本地 Windows/CPU 只能检查配置、源码和测试契约；
CANN 编译、NPU 数值与图回放正确性、HTTP 时延必须在目标卡分别验证。
不重新量化模型，不修改码本格式、RHT、FP32 bias dot、FP8 scale/div 或 MatMul 累加顺序。
E/F/G 不默认叠加，H 保持不接入模型；原 ABC 默认行为保留。

| 候选 | 修改 | 模型开关 | 当前边界 |
| --- | --- | --- | --- |
| I | clamp/SwiGLU → resident select/sign 融合 | `--swiglu-mode fused_select_sign` | 仅 decoder graph，先原生探针再独立原路径模型验收 |
| J2 | 每工作项复用输入行，处理 2 个 256 列 chunk | `--activation-reorder chunk_reuse2` | 仅 B1 decoder graph |
| J4 | 每工作项处理 4 个 chunk | `--activation-reorder chunk_reuse4` | 同上 |
| K | grouped work 从 route-major 改为 tile-major | `--b1-schedule tile_major` | 仅 B1 decoder graph，保持每 tile 的 K 次序 |

J 保留 route × chunk-group 并行度，避免 F 整行复用只有少数工作项的问题；K 仅改变工作项布局。
两者均不保证更快：必须看完整 projection 与 serving 效果，不能把 DMA 次数减少当作收益。
J/K 的 eager reference 和所有 prefill（包括 M=1 的 eager 投影）仍调用原 `project_vectorized`。
旧 F 的派发保持原样。J/K 与 D tail、scalar/row_reuse 混搭会明确报错。
缺失所选新 ABI/方法直接失败，不静默回退。

I 保留 BF16 clamp、SiLU、乘法的舍入边界，必须先通过同设备逐位原生探针。
独立探针的 `model_dispatch_enabled=false` 保持不变：它没有运行模型，不能冒充模型验收。
通过后才显式开启 I 的模型候选。eager/prefill 仍执行原 Torch SwiGLU 与原 select/sign 路径，
不能用同一融合实现同时充当候选与 reference。FAIL 时不得用 allclose 放宽原生逐位门槛。
I 限定 V4 v2、device-route decoder graph、`select-sign=fused`、`sign_fused_direct` preparation、
Torch tail、vectorized reorder、baseline B1 schedule；本轮不与 D/F/J/K 叠加。
默认 `--swiglu-mode torch` 完全保持原路径，缺失 I ABI/方法会直接报错，不静默回退。

## 1. 独立目录构建

若已经使用某个 `VQ2_IJK_LIB` 完成 I 原生探针验收，本次模型接线只改 Python，直接复用该 `.so`，
无需重新编译 CANN，也不重做 prepacked 权重。保留探针对应的库路径和 SHA256。
下面构建命令仅用于尚未拥有该库的首次构建，不要为已通过的库重复生成新路径。

沿用本机已确认的路径与型号。在内网 Linux 仓库和现有 torch_npu Python 环境中执行。
构建前先正常停止自己的 serving，确认物理卡 1 空闲；不要终止他人的进程。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
VQ2_CANN=/usr/local/Ascend/cann-9.1.0
VQ2_SOC=Ascend950DT_9582
VQ2_IJK_BUILD="$PWD/build/vq2a8-ascendc-v4-v2-ijk-$(date +%Y%m%d_%H%M%S)"
test ! -e "$VQ2_IJK_BUILD" || exit 1
VQ2_IJK_LIB="$VQ2_IJK_BUILD/libvq2a8_ascendc_v4_v2.so"

python -u tools/build_vq2a8_v4_v2.py \
  --soc "$VQ2_SOC" --cann "$VQ2_CANN" \
  --build-dir "$VQ2_IJK_BUILD" --jobs 4
test -f "$VQ2_IJK_LIB" || exit 1
printf 'VQ2_IJK_LIB=%s\n' "$VQ2_IJK_LIB"
```

必须得到 `V4_V2_BUILD=PASS`。不覆盖旧库，不重做 prepacked 权重。
后面 ABC 对照与候选使用同一新库；新开终端请恢复刚打印的完整 `VQ2_IJK_LIB`，不要重新生成时间戳。

## 2. 单项原生正确性

逐条执行，保存输出的 summary。FAIL/BLOCKED/超时均不算通过。

```bash
# I：独立数值实验，不会开启模型路径。
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_swiglu_select_sign.py \
  --library "$VQ2_IJK_LIB" --physical-npu 1 --queue-lifetime --timeout-s 1200

# J2 和 J4：分别验收，不能拿其中一个的结果替代另一个。
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_reorder_chunk_reuse.py \
  --library "$VQ2_IJK_LIB" --physical-npu 1 --chunks 2 --queue-lifetime --timeout-s 900

TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_reorder_chunk_reuse.py \
  --library "$VQ2_IJK_LIB" --physical-npu 1 --chunks 4 --queue-lifetime --timeout-s 900

# K：显式使用原 vectorized reorder，隔离调度变量。
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_b1_schedule.py \
  --library "$VQ2_IJK_LIB" --physical-npu 1 --reorder-chunks 0 --queue-lifetime --timeout-s 900
```

I 覆盖 BF16 位模式、不同 clamp limit、stride、路由/metadata 异常与恢复；
J 检查全部 FP8 raw codes（含 NaN/符号零）、descriptor、输出 poison、原投影对照；
K 检查保持逐位相同的投影与无效路由。
各探针还检查图回放、异步队列 owner 生命周期、普通操作与输入哨兵未被破坏。
I 数值失败不代表独立 J/K 失败，但不能略过 J/K 自己的验收。

## 3. I/J/K 实模型验证与时延

下面共享参数仅定义一次；每换阶段，重新执行 `case`，独立启动验证/服务进程。
建议先 ABC → I → ABC；J/K 仍分别按 ABC → J2 → ABC → J4 → ABC → K → ABC 测试，
减少温度/负载漂移误判。I 原生探针 PASS 不代替下面的实模型 validator PASS。

```bash
VQ2_COMMON=(
  --model /home/g00872988/vq2a8
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked
  --library "$VQ2_IJK_LIB" --physical-npu 1 --compute-backend v2
  --runtime-guard planned --validity-mode fused_vectorized
  --select-sign fused --activation-tail torch
  --activation-preparation sign_fused_direct --route-mapping fused
  --decoder-metadata-mode position_template --decoder-input-mode general
  --kv-cache-mib 256 --reserve-gib 3 --engine-memory-fraction 0.9
)

VQ2_STAGE=ABC
case "$VQ2_STAGE" in
  ABC) VQ2_MODES=(--swiglu-mode torch --activation-reorder vectorized --b1-schedule baseline) ;;
  I) VQ2_MODES=(--swiglu-mode fused_select_sign --activation-reorder vectorized --b1-schedule baseline) ;;
  J2) VQ2_MODES=(--swiglu-mode torch --activation-reorder chunk_reuse2 --b1-schedule baseline) ;;
  J4) VQ2_MODES=(--swiglu-mode torch --activation-reorder chunk_reuse4 --b1-schedule baseline) ;;
  K) VQ2_MODES=(--swiglu-mode torch --activation-reorder vectorized --b1-schedule tile_major) ;;
  *) printf 'Unknown stage: %s\n' "$VQ2_STAGE"; exit 1 ;;
esac

TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  "${VQ2_COMMON[@]}" "${VQ2_MODES[@]}" --timeout-s 1800
```

该阶段必须 `V4_DECODER_GRAPH=PASS` 后才启动以下服务。validator 内部固定 decoder 图、
caller stream、maxlen16，不接受下面 serving 专用参数。
新 receipt 要求模式相符、独立 vectorized reference 和 candidate graph 构建证据。
I 另须记录 `swiglu_mode=fused_select_sign`、
`swiglu_reference=deepseek_v4_swiglu_reference`，以及 `graph.swiglu_candidates` 中的路径证据。
每个 case 的 `swiglu_eager_reference` 必须证明原 Torch 路径有调用、eager 比较期间图关闭、
没有融合构图或 decoder replay；随后 candidate 的 replay 数必须等于该 case 输出 token 数减一。
模型比较保持原门槛：token/top-5 token ID 完全相同，logprob 使用既有 `abs_tol=1e-4`、
`rel_tol=1e-5`；这是模型级门槛，不代表全部 logits 逐位相同，也不能替代原生算子逐位探针。
`graph_build_calls` 是构图调用次数，不是 NPU replay 次数或性能证据；真实执行仍由硬件探针与模型比较确认。

```bash
TASK_QUEUE_ENABLE=1 python -u tools/serve_vq2a8_v4.py \
  "${VQ2_COMMON[@]}" "${VQ2_MODES[@]}" \
  --device-route-decode --decode-graph decoder --graph-replay-stream caller \
  --max-model-len 16 --memory-fraction 1 --port 8000
```

另一终端，保持原 4-token 口径，不开 profiler/host-profile：

```bash
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 --prompt '你好' \
  --max-tokens 4 --warmups 3 --repeats 50 --timeout 300
```

同时比较 TPOT 与 TTFT，至少做交错复测，不用单次最小值判赢。
若有候选收益，再 profiling 检查 I 的融合 SwiGLU/select/sign、prepare 的 2/4-chunk kernel 或 grouped_b1 命中，
以及 graph→scalar 时间是否下降；不把异步同步等待和 kernel time 重复相加。

## 4. 可选 J+K 组合

只有单项均通过且性能值得保留才测组合；不能直接叠加各自收益。
先给 `validate_vq2a8_b1_schedule.py` 传 `--reorder-chunks 2` 或 `4`，完整验收组合。
再分别用 `--activation-reorder chunk_reuse2/4 --b1-schedule tile_major`
运行 decoder validator 和 serving。组合 receipt 严格绑定 chunk 数，不接受 K-only 的报告代替。

回退只需回到 ABC flags 并重新启动，不需回滚权重或删除文件。
