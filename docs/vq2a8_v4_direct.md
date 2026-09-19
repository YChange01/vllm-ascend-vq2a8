# V4/v2：41 ms 基线之后的独立优化候选

保留已验证的 `sign_fused + planned`、原 `perf3` 库及预转换权重。
以下候选均需显式启用，默认值不变；本地 CPU 测试不能证明 NPU 正确性或 TPOT 收益。

## 改动范围

| 开关 | 改动 | 是否需要新算子库 |
| --- | --- | --- |
| `--decoder-metadata-mode planned_fast` | 启动时编译 metadata 检查动作；单次 update 内复用张量契约读取，验证全部通过后才复制 | 否，兼容 perf3 |
| `--activation-preparation sign_fused_strided` | sign 核直接读取 BF16/FP32、连续或展开/对齐行间距输入，省去输入的 Python FP32 转换和连续化 | 是 |
| `--activation-preparation sign_fused_direct` | 在 strided 基础上，逐专家 RHT 和 bias MatMul 用 `out=` 写入最终缓冲区，省去两次显式输出复制 | 是 |

新激活路径仍限 M=1、G=1..6、K=2048/4096。这里 G 是所选专家的行数，
不是权重量化 group size。Prefill 仍走原 rowwise 路径。
新 native 接口要求 `activation_sign_strided_version() == 1`；旧库不满足时明确报错，
不会回退或误报启用成功。权重格式和 resident bank ABI 不变，无需重新预转换。

不修改 RHT/bias 的 MatMul 输入形状和逐专家计算顺序，也不启用之前未通过位级校验的
完整 native quantizer。`out=` 可能改变 NPU 底层分发，所以单列为候选，必须逐字节验收。

`planned_fast` 保留结构、常量、CPU 值、指针、shape/stride/dtype/device、不可变
storage 和别名拓扑检查。不跨请求缓存验证结果，不移除 runtime contract 或最终有效性同步。
本次没有改写通用 model runner 的 `_prepare_inputs` 或 `_build_attention_metadata`。

## 1. 单独编译新库

在 Linux 仓库执行；若 CANN 安装位置不同，替换 `--cann`。使用独立目录，不覆盖 perf3。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023

python -u tools/build_vq2a8_v4_v2.py \
  --soc Ascend950DT_9582 \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-v2-direct \
  --jobs 4
```

这一步只编译，不证明设备执行正确。保留 build manifest 和日志。

## 2. 卡 1 上的激活验收

先停止自己的服务，确认卡空闲；不要重置卡或停止其他人的任务。
先验证 strided，再验证 direct，任意一步 FAIL/BLOCKED 都应停止并检查报告，不能放宽容差。

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_activation_packed.py \
  --library build/vq2a8-ascendc-v4-v2-direct/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 \
  --preparation-modes sign_fused_strided \
  --queue-lifetime \
  --timeout-s 900
```

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_activation_packed.py \
  --library build/vq2a8-ascendc-v4-v2-direct/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 \
  --preparation-modes sign_fused_direct \
  --queue-lifetime \
  --timeout-s 900
```

要求 `V4_PACKED_ACTIVATION=PASS`。验证包含原 rowwise 的 q/scale/bias 位级比较、
BF16/FP32、连续/展开/对齐行间距输入、图回放有效性恢复，以及输入所有者提前释放后的
队列生命周期。每个新模式的队列测试覆盖 12 种输入组合，各 513 次；不是性能基准。
BF16 最小正规数、次正规数和正负零另有原始位校验，并直接对比新旧 sign 算子的 FP32
输出，避免后续量化掩盖转换差异。
未指定 `--preparation-modes` 时仍只验证原 `rowwise_packed sign_fused` 两种模式。

## 3. 真实模型验收，继续复用预转换权重

每次只跑一个模型。下面验证组合候选；A/B 中使用的其他组合也应先通过此验收。

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library build/vq2a8-ascendc-v4-v2-direct/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 \
  --compute-backend v2 \
  --activation-reorder vectorized \
  --activation-preparation sign_fused_direct \
  --decoder-metadata-mode planned_fast \
  --kv-cache-mib 256 \
  --reserve-gib 3 \
  --timeout-s 1800
```

要求 `V4_DECODER_GRAPH=PASS`。这是同一候选后端的 eager/graph 比较，
不能替代上一步候选与原 rowwise 的位级比较，也不证明 TPOT 达标。

## 4. 分开测量收益

先保持旧库、`sign_fused` 不变，只切换 `planned → planned_fast`；该项不需要重新编译。
再用新库的 `sign_fused + planned` 复测编译基线，排除构建差异。
随后用同一新库分别测试 `sign_fused_strided + planned`、`sign_fused_direct + planned`，
最后测试 `sign_fused_direct + planned_fast`。每次只切换指定项，重启自己的服务。

组合候选服务命令（仅在前面验收通过后）：

```bash
TASK_QUEUE_ENABLE=1 python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library build/vq2a8-ascendc-v4-v2-direct/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 \
  --compute-backend v2 \
  --activation-reorder vectorized \
  --activation-preparation sign_fused_direct \
  --device-route-decode \
  --decode-graph decoder \
  --graph-replay-stream caller \
  --decoder-metadata-mode planned_fast \
  --max-model-len 16 \
  --kv-cache-mib 256 \
  --memory-fraction 1.0 \
  --engine-memory-fraction 0.9 \
  --reserve-gib 3 \
  --port 8000
```

新终端测量，不开启 profiler 或 `--host-profile`：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 \
  --model vq2a8 \
  --prompt '你好' \
  --max-tokens 4 \
  --warmups 3 \
  --repeats 20 \
  --timeout 300
```

记录每项配置、库 SHA256、TPOT/TTFT。需要诊断时再按
[host profiling 流程](vq2a8_v4_host_path.md#4-attribute-the-remaining-host-gap)
另开采集，输出到 `/home/g00872988/profiler_output` 下的新目录。
重点比较 Cast/复制任务数、激活相关任务和 `metadata_update`，不要把 profiler 的
replay 周期按比例换算成 HTTP TPOT，也不要把 scalar 等待与设备工作重复相加。

回退只需恢复旧 `perf3` 库与 `sign_fused + planned`，不删除或重建原权重。

## 本地验证边界

2026-09-19：相关 CPU 合约测试 701 项通过；全量 CPU 回归 4245 项通过、260 项跳过，
另有此前已记录的 14 项失败（源码 provenance、旧模型测试 fixture 和 Windows CPU
FP8/FMA 边界）。本次未放宽断言或绕过这些检查。

Ruff、Markdown 和 diff 检查通过；完整 `format.sh ci` 因本机未安装 `pre-commit`
无法运行。编译 dry-run、激活验证 plan-only 和真实 decoder 验证 plan-only 通过，
这些只验证执行计划，不代表原生编译或 NPU 验收。

本机无 CANN/NPU，新核编译、设备数值、图回放、队列生命周期和 TPOT 收益仍待上述卡 1 验证。
