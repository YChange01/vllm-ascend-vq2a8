# VQ2A8 E/F/G/H 独立验收与测时

本轮以已使用的 **ABC 配置**为对照，不把多个新开关同时打开。所有候选仍待内网
CANN 编译和真实 NPU 验收；本地 CPU 测试、源码审查、`--plan-only`/`--dry-run`
都不代表 NPU 正确性或性能通过。本说明不涉及提交或推送。

## 候选与边界

| 阶段 | runtime guard | activation reorder | decoder input | 说明 |
| --- | --- | --- | --- | --- |
| ABC | planned | vectorized | general | 本轮统一对照 |
| E-only | native | vectorized | general | 每步保留字段与 owner 检查，合并 native 张量元数据检查 |
| F-only | planned | row_reuse | general | 重排 kernel 按行复用输入读取；保留原 FP8 字节及描述符语义 |
| G-only | planned | vectorized | b1_packed | 合并 B1 block-table 上传及多 KV 组 slot mapping |
| H | 不接入模型 | 不接入模型 | 不接入模型 | bias dot 的独立逐位等价实验，允许以 FAIL 结束 |

其余保持：`validity=fused_vectorized`、`select_sign=fused`、`activation_tail=torch`、
`activation_preparation=sign_fused_direct`、`route_mapping=fused`、设备路由、decoder 图、
caller stream、`decoder_metadata_mode=position_template`。

G **不是重写整个 `_prepare_inputs`**。原始输入准备、CPU 镜像、token 上传、请求记账
继续执行；仅受限 B1 同步 decode 中的两处 block-table 操作被批处理。每步读取 live
KV 行与当前位置，固定输出地址，不缓存上一个请求/位置的物理 block 或 slot。
prefill、非 B1、spec/async/CP、多模态等不适用条件走原路径。

## 1. 独立构建目录，保留已有库

在内网 Linux 容器的仓库执行。无需卸载或重装 `vllm-ascend`，无需重新生成专家
prepacked 权重，也无需清理整个 `build/` 或重编 SAS/QLI。
当前 Python 的 editable 安装应指向该工作树。

先设置实际 CANN 路径和精确 SoC。此前用户设备日志中的型号为
`Ascend950DT_9582`；仍须与本机安装的 CANN 平台配置核对，不能只写 `Ascend950`。
下面有意将这两个值设为必填变量，避免假定安装位置。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023

# 先按本机实际安装填写：
# export VQ2_CANN=/实际/CANN/安装目录
# export VQ2_SOC=实际精确Ascend950型号
: "${VQ2_CANN:?先设置实际 CANN 安装目录 VQ2_CANN}"
: "${VQ2_SOC:?先设置实际精确 SoC VQ2_SOC}"
test -d "$VQ2_CANN" || exit 1

VQ2_EFGH_BUILD="$PWD/build/vq2a8-ascendc-v4-v2-efgh-$(date +%Y%m%d_%H%M%S)"
test ! -e "$VQ2_EFGH_BUILD" || exit 1
VQ2_EFGH_LIB="$VQ2_EFGH_BUILD/libvq2a8_ascendc_v4_v2.so"

# 只打印构建命令及源码 hash，不创建构建产物：
python -u tools/build_vq2a8_v4_v2.py \
  --soc "$VQ2_SOC" --cann "$VQ2_CANN" \
  --build-dir "$VQ2_EFGH_BUILD" --jobs 4 --dry-run

# 确认无误后实际构建：
python -u tools/build_vq2a8_v4_v2.py \
  --soc "$VQ2_SOC" --cann "$VQ2_CANN" \
  --build-dir "$VQ2_EFGH_BUILD" --jobs 4
test -f "$VQ2_EFGH_LIB"
printf 'VQ2_EFGH_LIB=%s\n' "$VQ2_EFGH_LIB"
```

实际构建必须得到 `V4_V2_BUILD=PASS`。后续包括 ABC 对照在内，均使用上面这一份
新库，避免把不同二进制的差异归因于开关。新开终端时重新设置刚打印的
`VQ2_EFGH_LIB` 绝对路径，不要重新生成时间戳猜目录。

## 2. E/F/G/H 独立正确性

先正常停止自己启动的服务，确认物理卡 1 空闲。不要重置卡或终止他人的进程。
逐条执行并保存 `summary.json`/日志；FAIL、BLOCKED 或超时均应先调查，不能跳过。

### E：host guard，无 NPU 张量执行

```bash
python -u tools/validate_vq2a8_runtime_guard_native.py --library "$VQ2_EFGH_LIB"
```

优先复用第 1 节构建好的共享库，使用 CPU 张量检查 replacement、pointer、shape、
stride、offset、dtype、device、schema 和 owner 生命周期，不运行 NPU 张量计算。
要求输出 JSON 的 `status=PASS`、`native_host_verified=true`。

可选：如果只想在未安装 CANN 的 CPU 环境单独验收这份 host 源码，可执行：

```bash
python -u tools/validate_vq2a8_runtime_guard_native.py --build-cpu
```

可选命令用当前 PyTorch 的 C++ 工具链仅构建 `runtime_guard_binding.cpp`，需要可用
编译器/Ninja；构建失败不等于 NPU 故障。已有共享库时不必再运行它。
E host PASS 仍不代替下节 E-only decoder 验收。

### F：row-reuse 重排

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_reorder_row_reuse.py \
  --library "$VQ2_EFGH_LIB" --physical-npu 1 \
  --queue-lifetime --timeout-s 900
```

要求 `V4_REORDER_ROW_REUSE=PASS`。验收 FP8 字节重排、描述符、投影结果、图回放、
错误/恢复和队列 owner 生命周期；不改变 RHT、bias dot 或 scale 算法。
`row_reuse` 只在 M=1 路径启用，M>1 仍使用原 vectorized 路径。
它减少重复输入 DMA，但也减少 chunk 级并行度，因此不能预设一定有性能提升。

### G：B1 table-upload / grouped-slot

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_decoder_input_plan.py \
  --library "$VQ2_EFGH_LIB" --physical-npu 1 \
  --queue-lifetime --timeout-s 900
```

要求 `V4_DECODER_INPUT_PLAN=PASS`。包含不同 KV 组、非零 storage offset、跨 block、
A→B→A live KV 更新、padding/未写入存储哨兵、整数边界、非法 native contract、
防越界及恢复、513 次异步 owner-release 检查。G 输入准备发生在 decoder 图外，
因此这个 standalone gate 不宣称输入 kernel 的 graph capture 已验收。

### H：bias dot 实验，绝不自动启用

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_bias_dot_probe.py \
  --library "$VQ2_EFGH_LIB" --physical-npu 1 \
  --queue-lifetime --timeout-s 900
```

向量乘加/归约可能与 CANN MatMul 的 FP32 舍入顺序不同。
`V4_BIAS_DOT_PROBE=FAIL` 若是 bit mismatch，应保留该证据，不放宽误差；H 不进入
下节任何模型/serving 命令。即使该实验 PASS，也不能据此直接替换模型路径。
可选 `--inputs /实际/输入.pt` 仅用于已有真实 CPU FP32 `rotated`、`weight_bias`
张量样本，具体 schema 由探针校验。

## 3. ABC / E-only / F-only / G-only 实模型验收

以下模型和 artifact 路径沿用本次用户的现有安装。进入新终端须先设置第 1 节
实际生成的 `VQ2_EFGH_LIB`；不做权重重转换。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
: "${VQ2_EFGH_LIB:?设置第 1 节实际生成的共享库绝对路径}"
test -f "$VQ2_EFGH_LIB" || exit 1

VQ2_COMMON=(
  --model /home/g00872988/vq2a8
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked
  --library "$VQ2_EFGH_LIB"
  --physical-npu 1 --compute-backend v2
  --activation-preparation sign_fused_direct
  --validity-mode fused_vectorized --select-sign fused --activation-tail torch
  --route-mapping fused --decoder-metadata-mode position_template
  --kv-cache-mib 256 --reserve-gib 3 --engine-memory-fraction 0.9
)

# 每轮只修改这一个阶段名，并重新执行 case：
VQ2_STAGE=ABC
case "$VQ2_STAGE" in
  ABC)    VQ2_MODES=(--runtime-guard planned --activation-reorder vectorized --decoder-input-mode general) ;;
  E-only) VQ2_MODES=(--runtime-guard native --activation-reorder vectorized --decoder-input-mode general) ;;
  F-only) VQ2_MODES=(--runtime-guard planned --activation-reorder row_reuse --decoder-input-mode general) ;;
  G-only) VQ2_MODES=(--runtime-guard planned --activation-reorder vectorized --decoder-input-mode b1_packed) ;;
  *) printf '未知阶段: %s\n' "$VQ2_STAGE"; return 1 2>/dev/null || exit 1 ;;
esac
printf 'STAGE=%s LIBRARY=%s\n' "$VQ2_STAGE" "$VQ2_EFGH_LIB"

TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  "${VQ2_COMMON[@]}" "${VQ2_MODES[@]}" --timeout-s 1800
```

依次为 **ABC、E-only、F-only、G-only** 执行该命令。每次均要求
`V4_DECODER_GRAPH=PASS`，同时保留 summary 中的实际模式和执行计数，不能只看进程退出。
decoder 验证工具内部固定设备路由、decoder 图、caller stream、最大长度 16；
不要给它添加仅 serving parser 支持的 `--device-route-decode`、`--decode-graph` 等参数。

G 除 eager/general 参考之外，还执行同一 decoder 图的
**general → b1_packed → general，关闭 template shadow** 对比，检查输出与实际
fastpath/copy/slot 命中计数。这样不会让参考与候选同时走 G 而掩盖错误。
无命中、未完成 no-shadow 验收或模式不符，都不能放行到测时。

## 4. 单开候选测 HTTP 时延

正确性通过后，在同一终端使用已选阶段的 `VQ2_COMMON`、`VQ2_MODES` 启动：

```bash
TASK_QUEUE_ENABLE=1 python -u tools/serve_vq2a8_v4.py \
  "${VQ2_COMMON[@]}" "${VQ2_MODES[@]}" \
  --device-route-decode --decode-graph decoder --graph-replay-stream caller \
  --max-model-len 16 --memory-fraction 1.0 --port 8000
```

服务 ready 后，在另一个终端运行原有 benchmark。URL 必须是纯文本 URL，不能
粘贴 Markdown 链接。不要增加 `--host-profile` 或 `--profile-dir`，也不要启动
`/start_profile`；正确性 gate 的 shadow 验证不用于计时。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --prompt '你好' --max-tokens 4 --warmups 3 --repeats 20 --timeout 300
```

建议运行顺序为 `ABC → E-only → ABC → F-only → ABC → G-only → ABC`，每轮正常
停止当前服务并新建进程，保持同一卡、同库、同权重、同请求与环境。全部请求应为
`TOKENS=4`；使用全部样本报告 TTFT/TPOT 的均值、中位数和尾部，不挑最快几次。
差异接近噪声时对双方等量增加 repeats。确认独立收益后再讨论组合，不预先宣称
某个阶段会达到目标 TPOT。

## 5. 无设备 dry run 与回退

只核对参数而不运行真实设备时，给 decoder/各 standalone 命令追加
`--plan-only`，给 serving/build 命令追加 `--dry-run`。decoder/standalone 的
`--plan-only` 和 build 的 `--dry-run` 可使用明确的占位路径，不生成硬件 PASS。
**serving 的 `--dry-run` 不同：模型和 artifact 目录、指定名称的 `.so` 文件仍必须
存在，且会读取库文件计算 hash**；它只是不启动 vLLM/NPU，不能拿不存在的路径运行。
这些检查均不能证明权重内容、native ABI 或真实 CANN 执行正确。

任一候选失败或退化，切回本文件的 ABC 行：`planned / vectorized / general`。
保持 `activation_tail=torch`，H 不进模型。回退不需要删除新库或已有 profiler 文件。
需要新的 profiling 时独立重启采集进程并另存目录，将 profiling 的 replay 周期、
device-task union 与无 profiling 的 HTTP TPOT 分开报告，不相加也不按比例缩放。
