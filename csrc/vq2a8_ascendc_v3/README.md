# VQ2A8 AscendC v3：常驻调度、v2 计算内核与 MoE decode 图

执行策略仍为 `ascendc_v3`，使用独立的 `libvq2a8_ascendc_v3.so`；不改 v1/v2 或其他后端默认值。
本轮把 v2 的 register pair-LUT 计算接入 v3 的设备常驻运行时，并增加融合准备和 MoE 图执行选项。
**20 ms TPOT 是待测目标；本地 CPU 验证不能证明 NPU 正确性或性能达标。**

## 快速服务测速

日常迭代使用这两个轻量入口。启动器直接执行标准 `vllm serve`，模型保持加载；
无需 acceptance report、build manifest 或 preflight 回执，也不会自动运行验收。
使用当前仓库代码和已编译的 v3 库，在服务器运行：

```bash
python -u tools/serve_vq2a8_v3.py --model /path/to/vq2a8
```

默认库为 `build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so`，卡 0、端口 8000、
`preparation=eager`、`decode_graph=none`；可用 `--library`、`--physical-npu`、`--port` 修改。
可加 `--preparation fused --decode-graph moe` 测融合和 MoE 图。
服务模式关闭逐层同步计时与诊断日志，并允许 vLLM 启动时的 dummy profiling。
默认 engine/cache 比例为 0.98/1.0、reserve 为 3 GiB，是下文说明的紧预算候选；
可用 `--engine-memory-fraction`、`--memory-fraction`、`--reserve-gib` 调整。

服务就绪后，在服务器的另一个终端执行：

```bash
python tools/benchmark_vq2a8_serving.py
# 需要多测几次时：
python tools/benchmark_vq2a8_serving.py --max-tokens 64 --repeats 3
```

客户端默认发一个预热请求，再测一个请求；通过 `/v1/completions` 流式生成 32 tokens，
输出 `TTFT_MS`、`TPOT_MS`、`TOKENS`。`--warmups 0` 可查看首次请求延迟。
TTFT 从发送请求计至首个非空生成内容；TPOT 为首末内容间隔除以 `usage.completion_tokens - 1`，
不把 SSE chunk 数当作 token 数。这是客户端 HTTP 时间，包含传输和输出缓冲影响。
`--prompt` 可替换短提示词；当前模型仍限定 prompt 加 output 不超过 128 tokens。
服务可保持运行并反复测速；改内核需增量编译并重启服务，改 Python 只需重启服务。
已有构建目录时，内核迭代只需：

```bash
cmake --build build/vq2a8-ascendc-v3 --target vq2a8_ascendc_v3 -j4
```

首次构建可用 `python tools/build_vq2a8_ascendc_v3.py --soc Ascend950DT_9574`，按实际芯片和 CANN 路径调整。
本地服务相关 CPU 配置、隔离模型方法、流解析及旧路径回归共 284 passed；
测试未加载真实 vLLM/NPU 引擎，服务启动仍待内网验证。

## 实现与开关

| 部分 | 当前实现 |
| --- | --- |
| 权重 | 启动时将原制品转成 v2 的 K 排序、packed zN 与 pair-LUT，直接写最终常驻 bank |
| 投影 | N128/K1024 分块、每个 AIV K512、Mmad K256、M 按 16 对齐、多级流水 |
| decode 调度 | 设备选择专家指针及元数据；固定 9-word 描述符和输入/输出工作区；无路由 ID 回传或描述符 H2D |
| `--preparation eager`（默认） | 原逐行 dense RHT、bias GEMV，然后量化和字节 permutation |
| `--preparation fused` | 保留上述 RHT/GEMV；原生融合 weight scale 乘法、amax、行 scale、归一化/clamp、FP8 cast 和 permutation |
| `--decode-graph none`（默认） | eager 提交 |
| `--decode-graph moe` | 每层捕获完整 B1 MoE：路由、准备、两次投影、SwiGLU、加权归约和共享专家 |
| prefill | B>1 保持 host-routed eager，投影使用同一套已转换的常驻权重和 v2 计算内核 |

读取的仍是 `experts_vq_ascend_v2` 制品，无需重新量化或改磁盘文件。
运行时通过已有 `convert_expert_payload` 做一次 CPU 转换，不保留另一份长期设备权重。
任意合法 `16×2` FP8 向量码本仍按 VQ 解释，不当成整数 INT4 或标量 FP4。
新的 resident 投影范围是 N4096、K2048/4096、M1..32、1..6 jobs，硬件目标为 Ascend950。
旧 v3 内核和入口留作诊断，正式 `ascendc_v3` 配置默认走新 resident ABI 1。

图捕获前在拥有工作区的同一 stream 上 warmup 两次。回放更新 hidden 和 token ID，
路由及指针选择在图内执行；输出复制到独立存储，避免下一 token 覆盖已返回结果。
图只接受一个 B1 输入签名和一个 stream；捕获/回放失败直接报错，不静默回退。
累计设备有效性标志保持原地址，图回放继续累积无效输入/输出状态，在验收边界检查。

**图范围是完整 MoE 层，不是完整模型。** Attention、KV 更新、MoE 外的根线性层及采样仍 eager，
报告始终为 `full_model_graph_verified=false`。RHT/GEMV 未融合，准备和路由仍有 PyTorch 中间张量；
图内临时张量、算子 scratch、graph pool 和内存碎片也需要预算预留。

## 数值与验收

K 排序和 v2 的 K256 累加会改变投影归约顺序，不能预设与 v1 逐 bit 相同。
默认 `--baseline-mode observe` 运行独立 v1 对照，记录逐 step logits 误差、token 差异及前缀是否相同，
不把观测报告当成模型质量通过。前缀分歧后的 logits 不能作为相同输入的数值对照。
需要原来的严格门禁时显式使用 `--baseline-mode exact`。
`--v3-only` 跳过 v1，报告 `baseline_comparison=not_requested`、`baseline_exact=null`。
三种模式都不会把 v3 自身重复一致冒充 v1 一致。

算子 preflight 包括 28 个投影案例，覆盖新 eager/out 入口、M 边界、1/6 jobs、两种 K、
真实转换权重、重复运行、非默认 stream 和原址更新 descriptor 指向的专家。
`--preparation fused` 还强制执行三类准备测试：复用与分块边界、FP8 midpoint/负零、非有限值与非法 int64 order。
融合准备必须与 eager 的 FP8 字节、FP32 行 scale 和 bias 位级一致，不能复用 eager 模式的预检回执。
投影沿用 v2 的独立数学 oracle 与数值门限；这不等同于全模型质量验收。

新构建 manifest/回执使用 schema 2，并校验源码、库 SHA256、resident ABI 与 capability。
旧 v3 `.so` 必须重编，不能只替换 Python 文件后复用旧回执。
原生 `grouped_projection_resident_out` 是可信内部 ABI：`int64[jobs,9]` 存放
`x, scale, bias, packed_zn, pair_lut, output, M, N, K`。
它依赖运行时持有的不可变权重 bank 和有界选择，不接受外部文件/RPC 的任意设备指针。

## 内存预算

全层预算在加载根权重后、分配专家 bank 前检查；不足时拒绝，不减少专家数或回退 LRU。
计划按每个独立分配取整，包括转换权重、int64 activation order、固定准备/投影工作区、H128 和有效性标志。

| 参数 | 用途 |
| --- | --- |
| `--engine-memory-fraction`，默认 0.98 | vLLM 启动空闲内存检查 |
| `--memory-fraction` / `--cache-memory-fraction`，默认 0.9 | 常驻权重和工作区可用比例 |
| `--cache-reserve-gib`，默认 16 | 为 KV、临时张量、graph pool 和运行峰值预留 |
| `--cache-budget-gib 0` | 根据当前可用内存计算预算，非无限预算 |

可用预算为
`max(0, min(free + max(0, reserved - allocated), total × cache_fraction - allocated) - reserve)`。
工具显式分配 1 GiB KV，必须包含在 reserve 内；reserve 不会被自动缩小。
手动 KV 配置下，engine 比例不是整个进程的显存硬上限。

以仓库中的 43 层几何（前 3 层各 1 expert，其余每层 256 experts）静态计算：

- 转换后权重和元数据：66,520,172,544 bytes。
- 固定工作区：61,252,096 bytes。
- 总常驻计划：66,581,424,640 bytes，约 **62.009 GiB**，不含根权重、KV、临时张量或 graph pool。

相较原 v3，activation order 由 uint8 tile ID 换为 int64，工作区也增加。
套用此前服务器记录的总容量 86,067,118,080 bytes、根权重分配 15,909,779,456 bytes，
即使 cache 比例为 0.99、reserve 为 3 GiB，仍短缺 505,982,669 bytes（约 482.54 MiB）。
这些是静态预算回归数据，不是本轮 NPU 测量。
下方 `1.0 / 3 GiB` 是显式紧预算候选；需检查实际峰值，尤其启用 graph 时。
`V3_RESIDENCY_BUDGET fits=true` 只表示计划通过，不能证明运行峰值装得下。

## 独立引擎快测

已有编译好的 v3 库时，运行独立快测入口，**不需要先跑完整验收**：

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/quick_benchmark_vq2a8_v3.py \
  --model /home/g00872988/vq2a8 --physical-npu 0
```

默认使用 `build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so`，只创建一次 `vllm.LLM`，
预热 1 次、测量 3 次，每次输入 10、输出 32 个 token。直接通过 `llm.llm_engine.step()`
运行完整模型和调度器，不是单算子微基准，也不启动 HTTP 服务。
当前这个入口固定使用 `preparation=eager`、`decode_graph=none`，不提供融合/图模式开关。
每轮在终端打印 TTFT/TPOT/E2E；最后看 `QUICK_V3_DONE` 的 `TPOT_MEDIAN_MS`、最小值和最大值。
预热不计入统计；若只想尽快看一次结果，可加 `--repeats 1`，但单样本波动较大。
输入、输出和预热分别通过 `--prompt-tokens`、`--output-tokens`、`--warmups` 调整；总长度不超过 128。

TPOT 为 `(末 token 返回时间 - 首 token 返回时间) / (输出数 - 1)`，
用主机时钟记录引擎返回 token 的时间，包含调度/提交开销，不是纯 NPU 内核时间。
计时循环没有逐 token 日志、额外同步或 NPU event；请求边界同步。
启动、预热和最终检查不计入 TPOT，`E2E_S` 包含该请求的首 token 等待及末尾同步。

快测跳过自动构建、算子/QLI/SAS preflight、旧库对照、重复 logits 比较、profiler 和报告目录生成。
仍保留库/源码身份、硬件与 SoC 匹配、显存预算和原有完整权重加载检查，结束时检查有限值和 43 层
resident decode 执行。它只输出速度估计，始终标记 `ACCEPTANCE=NOT_RUN`，不证明数值正确或质量合格。
如果修改了原生源码，需要先重新编译 v3；快测不会悄悄测旧 `.so`。

为减少启动预跑，两项上下文/批 token 上限均设置为本次输入加输出长度，默认 42，而非完整测速的 128。
因此不同长度间、与完整验收间比较时需要注意配置差异；同配置连续测量更适合判断优化趋势。
模型全量加载和引擎自身的启动预跑仍保留，不能承诺整个命令几秒完成。

该快测入口采用当前 80 GiB 服务器使用的紧预算默认值：
`--engine-memory-fraction 0.98 --cache-memory-fraction 1.0 --cache-reserve-gib 3`，
固定 KV 为 1 GiB，包含在 reserve 中；仍按实际空闲显存检查，不保证装得下，不自动降低 reserve。
原有完整验收脚本及其默认配置不变。

## 内网验证步骤

先构建并只做算子预检，不加载全模型。路径替换为内网实际位置：

```bash
python -u tools/accept_vq2a8_ascendc_v3.py \
  --model /path/to/vq2a8 \
  --soc Ascend950DT_9574 --physical-npu 0 \
  --preparation fused --preflight-only --jobs 4 --timeout 1800
```

随后依次测三组：默认 `eager/none`、`fused/none`、`fused/moe`，区分计算核、准备融合和图执行的收益。
以下是第三组，仅测 v3 的热态 TPOT；前两组分别调整两个开关：

```bash
python -u tools/accept_vq2a8_ascendc_v3.py \
  --model /path/to/vq2a8 \
  --library build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so \
  --physical-npu 0 \
  --engine-memory-fraction 0.98 \
  --memory-fraction 1.0 --cache-reserve-gib 3 \
  --preparation fused --decode-graph moe \
  --v3-only --benchmark --cases 10:32 \
  --warmups 2 --repeats 5 --progress-interval 5 \
  --target-tpot-ms 20 --timeout 3600
```

`--library` 跳过编译，仍校验身份并重跑算子 preflight。
需要 v1 数值观察时，去掉 `--v3-only`，添加
`--baseline-library build/vq2a8-ascendc-v026/libvq2a8_ascendc.so`；严格对照再添加 `--baseline-mode exact`。
v1/v3 分进程加载。候选引擎先做重复性/有效性和执行覆盖检查，再 warmup、正式计时。
图模式要求正式样本没有新增 capture，并且每次 decode 都有对应 replay。

`10:32` 有 31 个 decode 间隔；进一步可用 `--cases 10:64,32:64 --repeats 20`。
这仍是 TP1/B1、总上下文不超过 128 的离线实验。
启动/转换/常驻加载与图捕获均不计入热态 TPOT。
加 `--profile` 会另采集一次 CPU/NPU trace，不混入计时样本。
加 `--plan-only` 仅打印命令，不启动 NPU；`--timeout` 是每个子阶段的上限。

## 报告

`reports/vq2a8-ascendc-v3-*/` 包含预检回执、日志、可选 v1 对照、
`performance/summary.json` / `summary.txt` 和顶层汇总。
报告记录准备模式、resident 布局/ABI、逻辑 decode/投影次数、prefill 次数、图 capture/replay 次数。
逻辑投影计数不包含首次捕获的 warmup；它不是 profiler 实测的 kernel 数。
TTFT、请求平均 TPOT 的中位数和逐 token 间隔 P95 分开报告；设备 event 包括 host 提交间隙。
`--target-tpot-ms 20` 依据实测请求平均 TPOT 中位数判定，未测时为 `null`，不代表每个 token 都小于 20 ms。

## 本地验证边界（2026-09-12）

本机有 CPU PyTorch、pytest 和主机 C++ 编译器，没有 vLLM、torch_npu、CANN 或 NPU。
CPU 数学/契约测试通过隔离 package 启动依赖运行；图测试使用显式 fake backend，
原生测试只编译可独立的布局/调度头文件，不编译 AscendC 设备内核。
已覆盖转换、FP8 准备字节、动态/重复专家选择、固定地址、预算、配置、图输入更新及失败处理、验收工具。
相关集成回归结果为 **873 passed、3 skipped**（跳过项均需 NPU），包含 v2、v3、准备、MoE、offline 和内存预算测试。
Ruff、主机 clang-format、Markdown lint、禁用导入和布尔上下文检查通过；
`bash format.sh ci` 在缺少 pre-commit 时退出，未完成完整 hook 链。
未连接内网设备，未验证 CANN 编译、NPU 数值、真实图回放或 TPOT。
