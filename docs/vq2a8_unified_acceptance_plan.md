# VQ2A8 第 2、4 项实施与统一实测

## 当前范围

2026-09-09：按用户最新要求，只执行第 2 项“性能优化与完整实测报告”和
第 4 项“原生 FP8 指令与片上解码取证”。本文件替代原来的四项计划。

不新增第 1 项跨版本差异定位/质量评测，也不新增第 3 项 HTTP 服务、
并发、长上下文和稳定性验收。优化需要的输入校验、算子回归和同条件
logits 精确对照仍保留；它们不代表完整模型质量已验证。

用户没有设速度门槛：`performance_target_met=null`。
测量完整不等于变快；正优化、负优化、缓存缺失、超时都会保留。

## 本轮实现

- 新入口：`tools/accept_vq2a8_release.py`。
- 性能子进程：`tools/benchmark_vq2a8_offline.py`。
- 报告契约：`tools/vq2a8_perf_report.py`。
- 指令/数据流索引与审核回执：`tools/vq2a8_native_review.py`。
- 仿真采集器增加 grouped 入口、持久目录，并默认使用构建清单 SOC。
- 保留前一轮 KV block-size 与本地版本后缀识别修复；不重新安装依赖。

本地只执行 CPU 测试与命令/API 检查，没有访问用户 NPU。
本轮不修改 C++ 内核，因此当前同源码、同 SOC 的库无需重新编译。
不覆盖库、权重、已有报告，也不修改设备签名策略或全局 CANN 配置。

### 性能改动

测量模式显式开启，普通离线短验收默认不变：

- 关闭逐层/逐专家控制台打印、逐投影计时同步和整份 logits 的热路径 CPU 拷贝。
- 最终 hidden/logits 的有限值标记在设备上合并，请求完成后统一检查。
- 保留路由/专家数学中的输入检查、阻塞 H2D，以及驱逐权重前的完成同步。
  因此不是“完全无同步”的实现，也不宣称所有 host wait 已消除。
- compact 激活准备把同 dtype 激活的多次逐行 FP32 转换合为一次；
  sign 有效性比较不再先转 int16。单投影多行准备也可共享一次有效性决定。
- RHT 与 bias 的矩阵运算仍逐行执行，归约形状、E4M3FN 舍入、
  scale/bias 顺序及重复路由混合顺序不变。不同 dtype 的输入保留旧转换顺序。
- compact 不缓存激活有效性或可变权重校验结果。

没有盲目加入异步 H2D、共享 workspace/descriptor 复用或新的 tile 双缓冲。
这些改动需要这次实测/trace 证明热点，并先解决所有权与完成事件，不能冒充已经完成的优化。

### 测量口径

范围仍为严格有界 TP1 offline：BF16 roots、AscendC 专家、eager、
同步调度、单请求、总上下文最多 128 tokens。

默认三个输入/输出长度组合：10/4、32/32、96/32。
每组 baseline 与 compact 各两次预热、五次实测，按 AB/BA 交替顺序运行。
这是原有支持边界内的测量矩阵，不是服务并发矩阵；不测试 512/2048 上下文。

- baseline 与 compact 使用同一进程、同一模型、同一库及缓存预算。
  比较的是准备路径改动，不是不同库，也不使用旧服务器诊断日志作为速度分母。
- 模型加载前进行小型实机准备逐位对照；每个负载实测前保存
  baseline/compact 的诊断 logits，要求精确一致，不放宽容差。
- 实测保留实际 token 到达时间、有限值检查、生成 tokens 对照及每步 forward 数。
- 输出离线 TTFT、TPOT、端到端延迟、output tokens/s、NPU event 时间跨度；
  event 跨度包含主机提交间隙，不等于所有内核耗时之和。
- 输出首次请求、预热、每次实测原始样本、启动/加载时间、
  host RSS/HWM、HBM allocated/reserved/peak、设备空闲/总内存、
  expert payload H2D 字节、缓存命中/加载/驱逐、逻辑投影与启动数。
- prepare/projection 的主机分段计时标为包含提交/校验等待，不能当 kernel-only 耗时。
- 首次请求发生在引擎 profile 之后，不称为完全空缓存；后续新负载也保留已有缓存。
  只有实测没有缺失/驱逐时才标为热驻留对照，否则明确包含缓存影响。
- 保留低样本量 p95 提示，不宣称可靠 p99。
- 强制输出指定 token 数用于可比负载，不用其结果判定自然生成质量。

## 原生指令与片上路径

对最终库顺序收集二进制、fused 仿真、grouped 仿真。
每个仿真最多五分钟，额外给予有界的日志解析时间；不自动延长至 45 分钟。
使用本机设备名核对构建 SOC，禁止默认沿用旧卡 957d。

fused 使用 M=17/N=32/K=512；grouped 使用 M=17 与 M=1 两个独立任务。
覆盖两个 Vector 半区、M 尾部和四次 K tile 迭代。每个入口只有一次 launch。

完整轨迹中的 `CUBE/MMAD/dtype:E4M3E4M3` 且正 call_count
可记为指令观察；标量 MADD、零调用、空地址反汇编、超时后的半截轨迹不算通过。

`native_fp8_instruction_observed` 与 `native_instruction_verified` 分开：
必须结合实际库、源代码位置和轨迹地址审核操作数及
packed/codebook → UB 解码 → L1 → L0 → Cube 的消费路径。
必须区分最终输出写 GM 与完整解码权重落 GM。只匹配到 UB→L1 不足以批准。

默认生成 `REVIEW_REQUIRED`，不自动伪造片上验证结果。
可由审核者提供 JSON 回执，包含 reviewer、最终库 SHA256，以及 fused/grouped 各自的：

- `fp8_operands`
- `packed_decode_in_ub`
- `ub_l1_l0_cube`
- `no_decoded_weight_gm_roundtrip`

每项包含 `accepted=true`、解释与 `references`；
引用包含报告内相对 path、sha256、明确 location。
数据流结论还须引用报告内固定副本 `kernel.cpp`。
回执校验来源/覆盖，不代替审核者对地址、流水依赖与数据语义的实际审查。

## 服务器执行

先确保本轮代码已经同步到服务器。下面是代码到位后的唯一实测命令：

```bash
python -u tools/accept_vq2a8_release.py --model /home/g00872988/vq2a8 --library build/vq2a8-ascendc-v026/libvq2a8_ascendc.so --physical-npu 0
```

无需重新 pip install、repack 或编译；如果源码/库 SOC 不匹配，预检会明确报错。
进度和摘要显示在终端，完整原始日志留在新建的 `reports/vq2a8-perf-native-时间戳/`。

只查看计划（不导入 NPU、不执行测试）：

```bash
python tools/accept_vq2a8_release.py --plan-only
```

最终读取 `REPORT=...` 下的 `summary.txt` 与 `summary.json`。
返回码 0 表示两项必需证据均通过，2 表示待审核/尚未齐备；
失败和缺失原因以报告为准。不能把所有子进程 exit=0 当总体 PASS。

相同代码、模型、库、设备、参数下继续未完成的阶段：

```bash
python -u tools/accept_vq2a8_release.py --model /home/g00872988/vq2a8 --library build/vq2a8-ascendc-v026/libvq2a8_ascendc.so --physical-npu 0 --resume 报告目录
```

resume 重新检查当前设备，复用字节哈希一致的成功阶段。
失败阶段创建新 attempt，保留旧失败；不覆盖原文件。
模型 payload 的大小/mtime 与元数据哈希只是身份线索，不冒充全量权重哈希扫描。
输入/源码/环境变化需新目录；不把旧结果继承到不同版本。

审核后仅重建摘要，不再运行设备负载：

```bash
python tools/accept_vq2a8_release.py --summarize 报告目录 --review-json 审核回执.json
```

## 验收状态

- 性能通过要求完整样本矩阵和优化必需的数值回归通过；速度目标始终为空。
- 底层证据缺失、超时或尚未审核时不计为通过。
- QUALITY/SERVING 为 NOT_REQUESTED，不再阻塞本轮第 2、4 项。
- 本地测试通过不等于 NPU 编译、真实耗时或底层指令已验证。
