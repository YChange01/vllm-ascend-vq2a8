# TP1 完整权重启动定位

当无权重组件测试通过、但整模加载后仍长时间没有输出时，可保留原模型和权重，
给真实执行路径加实例级诊断包装。这个模式会重新加载完整 TP1 权重，仍需容纳完整常驻权重及启动工作区。
仅限 V3、TP1、eager、`--decode-graph none`；不是修改算子实现，也不是整模正确性验收。

## 第一轮：保持异步，只记录阶段

确认自己的旧进程已停止、物理卡 1 可供本次任务使用。在原来的容器、Python 环境执行：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023
python -u tools/serve_vq2a8_v3.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_tp1_zn \
  --library ./build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so \
  --tensor-parallel-size 1 --physical-npu 1 --port 8000 \
  --preparation eager --decode-graph none --reserve-gib 3 \
  --startup-trace async
```

若本机模型/artifact 路径不同，只替换相应路径。无需重新 repack 或重新编译已有的算子。
诊断对象只在显式设置 `--startup-trace async` 或 `sync` 时安装，默认 `off` 不导入诊断模块。
保持 `measurement_mode`、路由、分块、数据类型和原返回值不变；不是关闭原 serving 分支。

权重加载结束后会打印独立报告目录 `/tmp/vq2-full-startup-...`。
每层记录 decoder、HCPre/HCPost、LayerNorm、Attention、MoE；MoE 内继续区分
路由、gather、gate/up、SwiGLU、down、slot write、shared/mix、activation preparation、native 和 stream 绑定。
详细事件写入 `events.jsonl`，每 30 秒的 Python 栈写入 `stacks.log`；终端仅保留关键边界和等待提示。
`READY` 表示包装安装完毕、尚未进入模型调用，不代表计算通过。
Python 栈只能显示 Python 入口，不能代替 CANN 内核/驱动栈。

若迟迟没有下一个阶段，保持进程运行，等至少 30 秒让栈落盘，另开同一容器的终端执行：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python tools/summarize_vq2a8_full_startup.py
```

摘要默认读取最新报告，也可在命令末尾传入打印出的完整报告目录。
它只读日志，不加载权重、不初始化 NPU、不终止进程；输出限制在少量事件和最近主线程栈。
把摘要发来即可，不必贴完整加载日志。
`DEEPEST_OPEN` 是当前最深的未完成阶段；`DEEPEST_ERROR` 保留异常向外传播前的最深失败阶段。

## 第二轮：需要更细设备边界时

将同一启动命令的末尾改为 `--startup-trace sync`。每个被包装阶段返回后会显式同步设备。
`BEGIN` 表示进入，`SUBMITTED` 表示原 Python/API 调用返回，`PASS` 表示该阶段退出。
**async 模式的 PASS 不代表设备已完成**；sync 模式的 PASS 才包含显式同步完成。
没有 SUBMITTED 可能是 Python/C++ 调用未返回，不能仅凭此断定 CPU 死循环。
在同步处阻塞也可能来自此前提交的工作，需结合嵌套阶段和 Python 栈判断。

同步会改变执行时序、降低速度，也可能使异步问题暂时不再出现；因此不能把
“sync 启动成功”解释为根因已修复。首次调用也可能含编译/初始化开销，不能直接按层数外推。
诊断日志本身亦有额外开销，两种模式都不能用于性能结论。

诊断不会自动超时杀进程、重置设备或操作其他作业；定时栈在模型 forward 之外也继续记录，
以便区分“尚未进入模型”“模型已返回，worker 仍在后续初始化”等情况。
Python 的延迟栈定时器是进程级资源，此模式需独占它，不要同时启动另一套 `faulthandler.dump_traceback_later`。
诊断会在该进程后续请求中继续生效。定位结束后停止自己的诊断服务，再不带
`--startup-trace` 正常启动，不要把诊断模式当常规服务性能设置。

## 本次开发验证边界

本地仅能验证 CPU 配置和包装契约，未在本地 NPU 重新加载模型；现场运行结果仍需确认。
新增 trace/summary 的 56 项 CPU 测试通过，相关配置/启动测试合计 198 项通过。
全量 VQ2 CPU 回归仍有两项本次未修改、此前已复现的 root FP8 舍入测试失败，
本次不放宽数值断言，也不把合成/CPU 契约通过视为整模通过。
本次 Python 文件通过 Ruff；仓库 `format.sh ci` 因本机缺少 `pre-commit` 未能运行。
