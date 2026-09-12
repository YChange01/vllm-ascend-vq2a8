# VQ2A8 AscendC v3：设备常驻 decode 候选

v3 从 `../vq2a8_ascendc`（v1）派生，不替换 v1/v2，不改默认后端。
独立执行策略为 `ascendc_v3`，独立库为 `libvq2a8_ascendc_v3.so`。
继续读取原 `experts_vq_ascend_v2` **权重制品格式**；该目录名不是算子 v2，
不需要重新量化、K 排序或重新打包权重。

## 本轮实现与边界

已实现：

- 加载全部不可驱逐的 packed 专家权重池，直接写入最终存储，无长期双份权重。
  加载前按所有层计算权重、元数据、固定工作区预算，空间不足直接拒绝，不回退 LRU。
- B1 decode 在设备上选择专家指针及 sign/scale/bias，不把路由 ID 搬回 CPU；
  复用设备描述符和投影输入/输出工作区，取消逐层索引与描述符 H2D。
- 准备阶段点算子按批执行；仍逐行执行原 dense RHT 和 bias GEMV，保持旧数值顺序。
  保留任意 `16×2` FP8 向量码本，不将 VQ 编码误当成整数 INT4 或标量 FP4。
- prepared 原生入口从常驻 GM 表 DMA 加载 6,656 个常量字，
  取代每个 AIV、每次 launch 的逐项 scalar 初始化。
- M1 激活直接按 NZ 行搬运；每个输出块只清零一次，第二个 AIV 不再搬入空激活或 Gather。
- 保留原 K128 归约、FP32 scale→bias→BF16、SwiGLU 舍入边界和确定性路由归约。
  初始化验证权重；执行时保留设备有效性标志，验收边界检查，不能只凭有限 logits 通过。
- 当前 stream 的原生输入/工作区有 allocator 生命周期记录，运行时拒绝跨 stream
  及递归复用。MoE 返回值有独立存储，不与下一个 token 的工作区别名。

**没有完成/没有验证的部分：**

- Cube 仍用 M32；M1 优化的是激活搬运，不是已经消除了 32 行填充计算。
- RHT/bias GEMV 尚未融合成新的原生准备算子；不默认启用改变归约顺序的 FWHT。
- 准备阶段仍有 PyTorch 中间分配；prefill 仍是 batched、host-routed eager 路径。
- 没有全模型 DecodeGraph，也没有将局部准备图标成全模型图。
- 本机只能做 CPU 契约/数学/脚本检查；CANN 编译、NPU 数值和性能由服务器实测。
  **20 ms 是观察目标，不是已实现的性能，也不是本轮可保证的结果。**

后续是否缩小 Cube M、融合准备/SwiGLU/归约、接入全模型图，需先根据本轮
严格对照及设备实测决定；不通过放宽数值验收来换取成功标签。

## 内存预算

启动检查与权重常驻预算使用两个独立参数：

- `--engine-memory-fraction`（默认 `0.98`）传给 vLLM 的 `gpu_memory_utilization`，
  引擎启动时检查空闲显存是否达到 `total × engine_fraction`，不绕过 worker 检查。
- `--memory-fraction` / `--cache-memory-fraction`（同一参数，默认 `0.9`）
  单独限制 packed 权重和固定工作区预算，不再传给引擎启动检查。
- `--cache-reserve-gib` 保留给后续 KV、临时张量和分配器余量，不自动降低。

预算在 BF16 根权重加载完成后计算。`--cache-budget-gib 0` 表示按可用内存计算，
不是无限预算。可用预算为
`max(0, min(free + max(0, reserved - allocated), total × cache_fraction - allocated) - reserve)`。
固定 v3 decode 工作区计入常驻需求；显式 cache budget 也不得超过此可用预算。
独立 cache 比例仅在显式指定 KV 字节数且其不超过 reserve 时启用，本工具固定为 1 GiB，
避免同时使用自动 KV 比例预算；reserve 中 KV 以外的空间还要容纳临时张量和运行峰值。
因此 engine 比例在此手动 KV 配置下不充当整个进程的显存硬上限；物理安全检查由独立预算及预留承担。
其他旧入口未指定独立 cache 比例时，仍沿用原来的引擎比例，不改变旧版默认行为。

此前服务器日志中的物理容量约 80.16 GiB、根权重已分配约 14.82 GiB、
完整 packed cache 约 61.54 GiB。再扣除 1 GiB KV 后算术余量仅约 2.80 GiB，
**不能据此保证完整常驻可成功**，还要计算工作区及运行峰值。
默认 0.9 使用比例和 16 GiB reserve 很可能主动拒绝该配置。
这代表策略预算不足，不等于证明物理内存绝对装不下。

旧命令把 `1.0` 同时用作引擎启动比例，会要求整卡 80.16 GiB 全部空闲，
即使日志显示空闲 79.41 GiB，也会在模型加载前退出。这不是模型 OOM。
按此前精确根权重分配数及模型几何计算，v3 常驻需求约 61.566 GiB；
只改为 `0.99` 并保留 3 GiB reserve，比例预算仍少约 29.43 MiB，不能作为可靠修复。

以下完整实测示例显式选择 `--engine-memory-fraction 0.98 --memory-fraction 1.0 --cache-reserve-gib 3`，
是一个需要核查运行峰值的紧预算候选，**不是自动默认值或装得下的承诺**。
请在空闲卡上执行；若预算/OOM 失败，保留日志，不要无条件继续减小 reserve。
启动前显示 `PERF_V3_MEMORY_CONFIG`；根权重加载后显示 `MODEL_CACHE_BUDGET`，
全驻留分配前显示 `V3_RESIDENCY_BUDGET` 的需求、可用预算、余量/缺口和 `fits`。
如果 `fits=false`，不会开始部分权重加载；如果 `fits=true`，也不代表运行峰值已验证。

## 服务器操作

### 只测当前 v3 的 TTFT/TPOT

已有 v3 库且只关心当前速度时，使用 `--v3-only --benchmark`。该模式不读取、
不重编、不运行 v1 对照库，也不要求 v1 reference report；默认严格对照模式仍保留。
算子 preflight、v3 自身两次逐 step 重复性/有限值/执行覆盖检查仍然执行。
报告标记 `baseline_comparison=not_requested`、`baseline_exact=null`，
终端显示 `BASELINE_EXACT=NOT_REQUESTED`，不能据此宣称与旧版一致或质量已验证。

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/accept_vq2a8_ascendc_v3.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so \
  --physical-npu 0 \
  --engine-memory-fraction 0.98 \
  --memory-fraction 1.0 --cache-reserve-gib 3 \
  --v3-only --benchmark --cases 10:32 \
  --warmups 2 --repeats 5 \
  --progress-interval 5 --target-tpot-ms 20 --timeout 3600
```

这使用已编译的 v3 库，仅重跑算子检查；首次编译则去掉 `--library`，改传
`--soc Ascend950DT_9574 --jobs 4`。`10:32` 每次提供 31 个 decode 间隔，
比 `10:4` 的 3 个间隔更适合观察 TPOT。不传 `--profile` 可省去额外的性能跟踪请求。
上述显式紧内存预算仍有前文所述 OOM 风险，不会自动减少预留空间。

进度包括：阶段编号/耗时、每层常驻权重加载量、当前用例/轮次、
限频的 token 完成数和距最近 token 的等待时间，以及请求结束后的 TTFT/TPOT/E2E。
`--progress-interval 5` 控制后台 token 进度间隔；设为 `0` 关闭该进度线程，
但保留阶段/用例结果。计时循环只更新 CPU 标量快照，不为日志逐 token 同步 NPU
或写终端；后台日志仍可能影响主机调度，报告保留该配置用于复测。

### 与 v1 严格对照

先只编译并做短算子检查，不加载全模型：

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/accept_vq2a8_ascendc_v3.py \
  --model /home/g00872988/vq2a8 \
  --soc Ascend950DT_9574 \
  --physical-npu 0 \
  --preflight-only \
  --jobs 4 --timeout 1800
```

通过后使用刚编好的 v3 库，与已有旧版库对照并测量短 case：

```bash
python -u tools/accept_vq2a8_ascendc_v3.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so \
  --baseline-library build/vq2a8-ascendc-v026/libvq2a8_ascendc.so \
  --physical-npu 0 \
  --engine-memory-fraction 0.98 \
  --memory-fraction 1.0 --cache-reserve-gib 3 \
  --benchmark --cases 10:4 --warmups 2 --repeats 5 \
  --target-tpot-ms 20 --timeout 3600
```

`--library` 跳过编译，但仍检查库/源码身份并执行算子 preflight。
v1 参考运行与 v3 运行在不同子进程，避免同时保留两份模型。
启用 benchmark 时，v3 同一引擎先做严格数值对照再计时，不重复加载 v3 模型。
首次全量专家载入属于 startup，不计入热态 TPOT；每层加载有进度输出。
`--timeout` 是每个子阶段上限，不是所有阶段的总上限。

`10:4` 为 10 个输入 token、4 个输出 token，仅有 3 个 decode 间隔。
短测通过后将参数改为 `--cases 10:64,32:64 --repeats 20` 做较长输出测量。
这仍是 TP1/B1、总上下文不超过 128 的离线测试，不是服务吞吐或质量验收。

需要定位剩余瓶颈时加 `--profile`：测速完成后，在同一 v3 引擎中额外采集一次
CPU/NPU trace，不把该次请求计入性能样本。跟踪包含 `v3_device_route`、
`v3_gate_up`、`v3_swiglu`、`v3_down` 范围；gate/down 包括准备及提交，
不是纯内核时间。`PROFILE_STATUS` 单独报告，不代表原生指令或全模型图已验证。

只查看将执行的步骤，不使用 NPU：在上述命令末尾加 `--plan-only`。
也可用 `--reference-report <本工具生成的v1-reference/summary.json>`
复用同模型、同代码、同设备、同配置、同 cases 的已验证参考；身份变更即拒绝复用。

## 报告与验收含义

报告位于 `reports/vq2a8-ascendc-v3-*/`：

- `preflight.json` / `preflight.log`：独立数值 oracle、原生 grouped/prepared 路径检查。
- `v1-reference/`：旧版逐 step logits、token 和源码/库/模型身份。
- `performance/summary.txt`、`performance/summary.json`：
  TTFT、请求平均 TPOT 中位数、逐 token 间隔 P95、设备 event 观察值和原始样本。
  event 跨度含 host 提交间隙，不是纯 Cube 内核耗时。
- 顶层 `summary.json`：各阶段结果及最终验收状态。

严格对照模式中的 `BASELINE_EXACT=True` 必须由 v1/v3 的逐 step FP32 logits 字节及输出 token 比较产生。
重复一致不代替 baseline 一致。热态样本要求全层执行、零专家换入/换出，
并保留 v3 设备路由计数。未获得性能数据时目标结果为 `null`。
`--v3-only` 中性能测量可以通过，但 `BASELINE_EXACT` 始终为 `NOT_REQUESTED`。
`--target-tpot-ms 20` 根据实测请求平均 TPOT 的中位数报告目标是否满足；
目标未满足不伪造功能失败，也不把 P50 达标冒充每个 token 均达标。

原生 `grouped_projection_out` 是**可信内部工作区 ABI**：
它接收设备上的指针记录，不能接受文件/RPC 提供的任意描述符。
binding 的 shape 检查并不能认证任意描述符中的指针；它依赖运行时预验证并持有的
不可变专家权重池和有界设备选择。不要把此入口作为通用对外算子接口。

## 本地验证记录（2026-09-11）

- CPU 全套回归：1,452 passed、213 skipped、2 failed（另有 4 subtests passed）。
- 两个失败均为此前已有的 root-FP8 midpoint/FMA CPU 测试；本轮未改该路径，v3 仅启用 BF16 根权重。
- Ruff、clang-format、Markdown 检查通过；`format.sh ci` 因本机缺少 pre-commit 未执行完整检查链。
- 未进行 CANN 编译、NPU 执行或 TPOT 验证；上述 CPU 结果不代替服务器验收。
