# VQ2A8 算子 v2

这是适配公司内部参考实现的独立候选后端，不是将原示例改名后直接替换。
原始聊天文本和恢复稿仍保存在 `../vq2a8_expert_reference/`；
逐文件核对见 [SOURCE_RECONCILIATION.md](../vq2a8_expert_reference/SOURCE_RECONCILIATION.md)。
原代码权利声明不变；上传仓库不等同于为原参考代码授予新的开源许可证。

## 状态与命名

- 执行策略：`ascendc_v2`；Torch 命名空间：`vq2a8_ascendc_v2`。
- 独立库：`libvq2a8_ascendc_v2.so`；初始接口版本：ABI 1。
- 构建、Python 缓存适配、模型选择和分阶段验收入口已经提供。
- `--benchmark` 显式增加独立进程的 v2 TTFT/TPOT 测量阶段，默认不启动计时。
- 本地 Windows 只有 CPU 环境；尚未在 CANN 上编译或在 Ascend950 上执行，不能宣称精度或性能通过。
- 原 `ascendc`、`libvq2a8_ascendc.so` 和服务默认设置保持不变，不使用旧库兜底。

## 保留与调整

| 部分 | v2 处理 |
| --- | --- |
| 寄存器查表 | 保留 `DIST_UNPACK4_B8` 与任意 16 项双 FP8 字节查表；不拟合成四标量 W2 |
| 内存流水 | 保留 zN、UB 每 K16 补一行、packed/decoded/LUT 双缓冲；N128、每 AIV K512、AIC K1024、MAD K256 |
| 真实权重布局 | 首次缓存装载时将 int32 打包转换为 uint8 zN，并按码本 ID 稳定重排 K；不修改磁盘权重 |
| 激活 | 原 RHT/sign/量化完成后，对 FP8 字节做同一 K 重排；不改成示例的 K32 MXFP8 量化 |
| 矩阵计算 | 使用有类型的普通 FP8 `Mmad`，不需要模拟任意 FP32 scale 的 E8M0 编码 |
| 输出 | 有类型的 FP32 Fixpipe → FP32 行 scale/bias → 最后一次 BF16 舍入；绕开原恢复稿输出配置位歧义 |
| 分组接口 | 1..6 个独立指针描述符，不要求连续权重池；一个 MIX launch 包含投影和输出修正 |
| 调度与流 | 动态 AIC 数、1C:2V；使用 Torch 当前 NPU stream 和分配器，输入保活与跨流记录 |
| 构建 | CANN 安装目录提供的 `ascendc_library` 负责 AIC/AIV/MIX 编译、链接和 launch stub |

K 重排在实数代数上保持点积，且转换保留权重与激活字节，**但会改变 FP32 累加顺序**。
不得沿用旧库 `baseline_exact` 的结论。v2 验收保留原逐投影容差，重复执行仍要求逐位一致；
整模型短执行通过也不等于与旧模型 logits 一致或质量通过。

## 当前边界

- 仅 Ascend950、TP1、BF16 根线性层、单请求 eager 离线模型，最多 128 context tokens。
- 原生单 job：`1 <= M <= 32`，`N=4096`，`K=2048` 或 `4096`；每组 1..6 个同 N/K job。
- 对应当前模型 gate/up 与 down 两路；更长输入由现有调度拆分，不是将任意形状送入原生算子。
- 支持 `fast` 和 `batched` 准备/调度预设。旧 `pipeline`、`fwht`、`prepare_graph` 不属于本次 v2 验收范围。
- 未实现更大 M/N6144 档位、整模型图、TP/EP、多请求并发或 HTTP 服务切换。
- `experts_vq_ascend_v2` 是既有**权重格式目录**，不是这里的新算子版本；无需再次 repack。

缓存只保存新压缩布局、pair LUT、激活 K 索引及原准备元数据；显存预算包含新增 int64 索引。
不在热路径拼接全部权重，不生成真实专家的设备端 dense 权重池。
仍有量化后的激活 gather，以及每组最多 432 字节的阻塞描述符 H2D；不能将其耗时排除后声称模型提速。

## 本地验证记录（2026-09-09）

环境：Windows、Python 3.14.7、Torch 2.10.0+cpu。CPU 测试使用仓库外的
`../migration-v026/run_cpu_tests.py` 隔离运行器：vLLM 使用接口桩，Triton 包装器禁止执行设备内核。
这些结果只验证 CPU 数值、布局和接口约束，不代表真实 vLLM/NPU 模型执行。

- v2 新增测试（含计时工具与报告校验）：239 通过、1 跳过（本机无 C++ 编译器）。
- v2 与相关构建、执行、离线、v0.26 和优化回归：777 通过、5 跳过。
- 原参考实现的 CPU unittest：63 通过。
- 全量 CPU 约束测试：1369 通过、212 跳过、2 失败；不是全量通过。
- 变更 Python 文件的 Ruff 检查/格式、原生文件的 clang-format 检查、`git diff --check` 通过。
- `bash format.sh ci` 因本机未安装 `pre-commit` 未完成；未据此声称 CI 通过。

两项失败均在未修改的 `test_vq2a8_root_fp8.py`：
`test_inverse_rope_fp8_midpoint_regression_and_bounded_diagnostics` 与
`test_rounding_probe_requires_exact_inputs_scales_and_fp32_before_weight_gates`。
使用同一 Python 环境，在干净的旧提交 `f989dec9b0c1a54e2609976b1b82bf2f5376f758` 快照中也复现了相同失败。
本轮未修改这些测试、根 FP8 实现或容差；v2 当前仅开放 BF16 根线性层。

构建 dry-run、验收 plan-only 和工具帮助入口已检查。CANN 编译、真实设备预检、整模型输出、
数值对照及 TTFT/TPOT 仍须在服务器验证，不能由上述 CPU 测试替代。

## CANN 9.1 移位类型修复（2026-09-11）

服务器首次构建在 AIV 的 `ShiftRight` / `ShiftLeft` 报错：`vshr` / `vshl` 的
uint32 数据操作数要求配套 **int32 移位位数**。已仅将 `shiftRight`、`shiftLeft`
及其初始化值改为 `int32_t`；数据、索引、掩码仍为 `uint32_t`，移位值仍为 4、16。
不改变查表算法、权重布局、ABI 或默认后端，无需 repack 或重装 Python 包。

新增两项源码类型回归检查在修复前均失败、修复后通过；v2 CPU 测试为 241 通过、1 跳过。
全量 CPU 约束回归为 1371 通过、212 跳过、2 失败，仍为上面记录的两项根 FP8 既有失败。
Ruff、clang-format、Markdown 和变更源码检查通过；完整格式检查仍因缺少 `pre-commit` 未完成。
这不等于 CANN 编译通过。拉取修复后应重新运行下面的完整构建/预检/模型/计时命令，
不要用 `--library` 跳过构建，也不要复用旧预检结果。

## 服务器一条命令构建并验收

先将本轮文件同步到已安装的同一仓库，在原来通过依赖检查的 Python 环境执行：

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/accept_vq2a8_ascendc_v2.py \
  --model /home/g00872988/vq2a8 \
  --soc Ascend950DT_9574 \
  --physical-npu 0 \
  --preset batched \
  --benchmark --cases 10:4 \
  --jobs 4 \
  --timeout 3600
```

顺序是环境 → 单独构建 → 24 个预检案例 → 两次整模型短生成 → 独立计时进程。
不加 `--benchmark` 则止于短模型验收。`10:4` 表示输入 10 token、输出 4 token；
短案例通过后可用 `--cases 10:4,32:32,96:32` 扩大测量范围。
任何阶段失败即停止，不重装 pip 包、不清理旧构建、不修改设备验签设置。
`--timeout` 是每个子阶段上限，不是预计耗时；首次 CANN 编译时间尚未实测。

预检含两个实际 K、M=1/2/3/15/16/17/31/32 边界、六 job、重复执行、分组/单 job 对照、
非默认 stream、layer3/expert0 两路真实权重以及 deterministic/zero/impulse 输入。
合成整数/二进制 scale 案例要求严格相等；真实案例使用现有 `rtol=0.01, atol=0.001` 与相对 L2 门禁，
不因 v2 失败自动放宽。CPU dense 权重仅用于预检 oracle，不进入模型设备路径。

只构建：

```bash
python -u tools/build_vq2a8_ascendc_v2.py \
  --soc Ascend950DT_9574 --build-dir build/vq2a8-ascendc-v2 --jobs 4
```

复用已构建的 v2 库重新验收：

```bash
python -u tools/accept_vq2a8_ascendc_v2.py \
  --model /home/g00872988/vq2a8 --physical-npu 0 \
  --library build/vq2a8-ascendc-v2/libvq2a8_ascendc_v2.so \
  --benchmark --cases 10:4
```

加 `--preflight-only` 只做算子预检；加 `--plan-only` 仅打印命令、不会写文件或操作设备。
构建工具另有 `--dry-run`、`--cann`、`--ascendc-cmake`，用于核对已安装工具链。

## TTFT/TPOT 测量口径

`--benchmark` 只测试所选 v2 后端，不调用旧的 `accept_vq2a8_optimizations.py`，也不将旧库数据当作本次基线。
正确性阶段保留调试同步；仅独立计时子进程设置 `ASCEND_LAUNCH_BLOCKING=0`。
每个案例先执行相同 v2 预设的两次未计时诊断，检查完整原生调用覆盖和 logits/tokens 重复一致，
然后单独记录首次计时请求、默认 2 次预热和 5 次正式测量。最终摘要只统计正式测量的中位数。

- TTFT：从本地引擎提交请求开始，到首次取得输出 token 的墙钟时间。
- TPOT：首 token 之后至最后 token 的时间，除以输出 token 数减一。
- E2E：完整请求至结束同步的墙钟时间；不含引擎启动和上述独立诊断。
- 首次计时请求发生在诊断之后，**不是冷缓存 TTFT**；引擎启动与诊断耗时另列。
- 记录逐 token 时间、原生调用数、内存及缓存装载/淘汰变化。正式样本出现 miss/eviction 时不能标记为热缓存结果。
- 这是 TP1 单请求离线延迟，不是 HTTP 客户端延迟或并发吞吐；无硬性速度门槛。

## 结果查看

验收输出目录是 `reports/vq2a8-ascendc-v2-<时间>/`，保留：

- `summary.txt` / `summary.json`：分阶段状态、作用范围，不把编译通过写成模型通过。
- `environment.log`、`build.log`、`preflight.log`、`model.log`：完整日志和失败回溯。
- 选择计时后另有 `performance.log`、`performance/summary.json` 和 `performance/summary.txt`：
  各案例 TTFT/TPOT/E2E、所有原始样本、预热记录和诊断证据。
- `preflight.json`：真实设备、SoC、当前库/源文件/模型身份及完整案例结果。
- `model-evidence/run-*.json` 和 logits safetensors：两轮 tokens、logits、原生调用覆盖及缓存证据。
- 构建目录 `build-manifest.json`、`configure.log`、`compile.log`：实际 SDK、SoC、源码哈希和命令。

只有 `MODEL_EXECUTION_VERIFIED=True` 才代表这次 v2 短模型执行通过。
`PERFORMANCE_MEASUREMENT_VERIFIED=True` 只代表本次延迟测量及其完整性检查通过，不代表达到提速目标。
目前验收不出具旧库逐位一致、原生指令反汇编、模型质量或 serving 通过结论。
先跑通此门禁，再用同输入、同热缓存、同测量边界做旧库/v2 性能与数值对照。
