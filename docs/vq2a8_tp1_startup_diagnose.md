# TP1 权重加载后卡住：无模型权重诊断

适用于 V3 全常驻 TP1 加载结束、进入 dummy/profile forward 后没有进度的定位。
此工具只生成合成小张量，不读取模型、repack artifact，不创建 LLM、KV cache 或 43 层模型。
它不改变服务代码，不自动编译，不重置卡，不终止其他作业。

## 运行

先停止自己的模型进程。在已经配置好 CANN、torch-npu 的容器和 Python 环境中执行：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023
python -u tools/diagnose_vq2a8_tp1_startup.py --physical-npu 1
```

不需要重新安装 Python 包或 repack。HC 测试依赖当前环境已编译的
`vllm_ascend.vllm_ascend_C`；投影测试默认使用
`build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so`，可用 `--library` 指定。
缺少对应库会在导入/加载阶段明确失败，不会偷偷改用其他实现。

默认按顺序运行：基础 BF16 加法/矩阵乘法，HCPre M2/M128，独立 HCPost M128，
真实 eager 激活准备 M32/六组 M32，V3 resident 投影 M1/M32/六组 M32。
六组代表六个专家投影任务，不是六层模型；准备阶段覆盖 K4096/K2048。
每项使用新子进程，默认每项 120 秒（含导入），首个失败或超时即停止。

每项前只读取所选物理卡的占用快照，使用可见设备限制映射到逻辑 NPU 0。
已占用或占用未知时默认拒绝执行。`--allow-busy` 只覆盖明确的占用，
不覆盖未知状态；这不是独占预留，共享卡仍可能影响其他作业。
合成张量远小于整模权重，但 NPU 运行时本身仍会占用显存。

## 读取结果

终端持续显示阶段，空闲 15 秒输出 heartbeat，子进程每 30 秒输出 Python 栈。
结果保存在启动时打印的 `REPORT=/tmp/vq2-tp1-startup-...`：

- `summary.json`：总状态、每项状态、最后阶段、子进程是否回收。
- `<case>.log`：该项完整日志、Python 栈、库路径/版本与校验信息。
- `<case>-npu.log`：该项开始前的设备占用快照。

阶段以 `STARTUP_PROBE` 开头：`BEGIN` 是调用前，`SUBMITTED` 是代码段已返回，
`PASS` 是显式 NPU 同步完成；若在 CPU 验证阶段失败，不应解释为算子死锁。
只有 `BEGIN` 而没有 `SUBMITTED` 时可能阻塞在调用内部；有 `SUBMITTED` 而无
`PASS` 时可能阻塞在同步。默认同步下发也可能使等待发生在调用内部。
这些结果用于缩小范围，不能单凭一个超时断定硬件故障或某条内核语句有错。
若最后已输出 `CASE_PASS` 却仍报告 `TIMEOUT`，说明探针主体已完成、进程退出没有完成，
应排查退出/析构，而非将该项误判为计算核超时。整项超时也包含 CPU 合成和验证。

超时仅向本次创建的子进程会话发送 TERM，最多等 5 秒再 KILL、等 3 秒。
回收进程不等于已取消设备任务或恢复设备状态；超时后不自动测试下一项。
不要反复重试处于异常状态的卡，也不要为了诊断重置其他人使用的设备。

可以单独选择一项，或提高包含导入在内的超时：

```bash
python -u tools/diagnose_vq2a8_tp1_startup.py --physical-npu 1 --cases hc_pre_m128 --timeout-s 180
```

`--launch-blocking 1` 是默认值，只作用于诊断子进程；可用 `--launch-blocking 0`
对比异步下发，但每个阶段仍显式同步，耗时不是推理性能。
`--plan-only` 仅展示执行计划，不导入 torch/vLLM、不访问设备、不创建报告。

全部 PASS 只说明这些独立合成用例通过，不等于整模启动、数值质量、组合算子路径或性能通过。
HC/准备阶段只验形状、类型和有限值等基础契约；V3 投影另验合成 FP8 oracle 的逐位一致性。
定位时请提供 `summary.json` 和首个失败/超时项的完整 `.log`，无需再次加载所有权重。

若这些组件全部通过，但整模仍卡住，可改用[完整权重启动定位](vq2a8_full_startup_trace.md)，
在一次真实加载后记录逐层阶段及定时 Python 栈。

## 本次开发验证边界

本地 Windows / CPU 验证了参数、日志、真实合成张量准备、错误路径、超时和设备隔离契约；
没有本机 NPU，因此实际 HC/原生投影执行仍待现场运行，不能把 CPU 测试视为设备通过。
全量 VQ2 CPU 测试运行时另有两项未改动的 root FP8 舍入测试失败，
以及一项计时测试首次失败、独立复跑通过；不在本次诊断工具中更改这些数值/计时断言。
本次 Python 文件通过 Ruff；仓库 `format.sh ci` 因本机缺少 `pre-commit` 未能运行。
