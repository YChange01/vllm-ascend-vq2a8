# 回到 V1 并复测 TPOT

这里的回退是选择本分支保留的 **`execution_policy=ascendc`**，仍使用 vLLM/Ascend 0.23。
不回滚整条分支，不改 V3 算法，不删除或覆盖已 repack 的 TP1/TP2 zN 权重。
直接复用现有编译和验收入口，不增加诊断脚本。

## 权重路径

沿用此前模型根目录 `/home/g00872988/vq2a8`。V1 读取：

- 根目录中的 `config.json`、`tokenizer.json` 和 root safetensors。
- `experts_vq_ascend_v2/` 中的旧 direct 专家，manifest 格式必须是 `vq2a8_direct_tp1_v1`。

目录名中的 `v2` **不是 AscendC V2 内核**。`experts_vq_tp1_zn`、`experts_vq_tp2_zn`
不能交给 V1，原始 `experts_vq` 也不能直接替代 direct artifact。只因切换框架版本无需重做旧 direct。
下面命令要求旧 direct 已存在；不要把新 zN 目录改名或修改 manifest 来绕过检查。

```bash
python -c 'import json; from pathlib import Path; p=Path("/home/g00872988/vq2a8/experts_vq_ascend_v2/manifest.json"); f=json.loads(p.read_text())["format"]; print(p, f); assert f == "vq2a8_direct_tp1_v1"'
```

这里只读取格式标识，不证明 payload 完整；正式入口仍执行完整性和数值预检。
若旧 direct 确实不存在，从原始 canonical 专家生成，输出目录必须未占用：

```bash
python tools/repack_vq2a8_tp1.py \
  --input /home/g00872988/vq2a8/experts_vq \
  --output /home/g00872988/vq2a8/experts_vq_ascend_v2 \
  --model-config /home/g00872988/vq2a8/config.json \
  --layers all
```

## 一条命令：编译 V1、加载一次模型、复测

在容器中使用已安装好 v0.23 主插件的同一个 Python，先正常退出自己的 V3 作业，
确认物理卡 1 空闲且可用。不要重置 NPU 或停止其他人的作业。
这是离线单请求测试，不启动 HTTP 服务。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023

python -u tools/accept_vq2a8_optimizations.py \
  --model /home/g00872988/vq2a8 \
  --soc Ascend950DT_9582 \
  --build-dir build/vq2a8-ascendc-v023-v1 \
  --jobs 4 --physical-npu 1 \
  --presets batched --cases 10:4 \
  --warmups 2 --repeats 5
```

`Ascend950DT_9582` 来自此前构建日志；**若当前卡型号不同，替换为当前卡准确的 SOC 名称**，
不能仅凭 `npu-smi` 概览中的 `Ascend950DT` 猜后缀。
构建使用 `ASCEND_HOME_PATH`，未设置时默认 `/usr/local/Ascend/cann-9.1.0`。
主插件的 HC/attention 算子必须已完成安装；此命令只编译独立 V1 专家库，不能替代主插件编译。

这个入口会依次检查实际运行时导入/API、编译 V1、运行小算子预检，再加载一次全模型进行 baseline/batched
数值比较和交替计时。使用预算内懒加载专家缓存，默认预留 16 GiB，不做 V3 全专家常驻；
不启用 V3 startup trace、fused preparation、MoE graph 或其他候选。
预检失败即停止，不加载全模型，也不引用旧机器 PASS 来跳过本次验证。

如需单独编译，或显式指定 CANN 路径：

```bash
python -u tools/build_vq2a8_ascendc.py \
  --soc Ascend950DT_9582 --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v023-v1 --jobs 4
```

输出是 `build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so`，不是 `libvq2a8_ascendc_v3.so`。
已经编好时，将复测命令的 `--soc`、`--build-dir`、`--jobs` 三项替换为
`--library build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so`，即可跳过构建。
不要把 V1 库传给 `serve_vq2a8_v3.py`；该入口固定使用 V3。

## 编译完成后仍停在 environment

按用户要求，已移除自动流程中的精确包版本、editable/Git 来源和全局 `pip check` 一致性门槛。
不再因 torch-npu 开发后缀、Ascend SCM 版本或镜像中 `affinity-sched` / `ms-service-profiler`
的全局依赖报告阻断复测；不需要补装这些包或添加跳过参数。
版本只记录，不比较，报告标记 `consistency_checked=false` 和 `pip_check_run=false`。

`environment` 阶段现在仅检查实际导入/API，后续设备、算子库/SoC、权重和数值检查保留。
真实运行需要的依赖若缺失，仍会在导入或执行时明确报错；这与全局一致性检查不同。

已经完成本机编译和 editable 安装时，更新脚本并复用现有 V1 库即可，不需要为本次工具修复重编译：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023
python -u tools/accept_vq2a8_optimizations.py --model /home/g00872988/vq2a8 --library build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so --physical-npu 1 --presets batched --cases 10:4 --warmups 2 --repeats 5
```

如果仍失败，`OPTIMIZATION_ERROR` 会显示有限长度、常见凭据脱敏后的实际环境错误；
`run.json` 也记录 `failure_causes`。无法识别的日志仍给出完整日志路径，不盲目回显原始日志。
真实导入失败和预检错误仍会停止。环境通过不等于 NPU 推理或 TPOT 通过。
严格一致性诊断只保留为手动工具，见[环境说明](vq2a8_v023_migration.md)，不再被自动复测调用。

## 0.3 秒的来源与结果

历史用户回传报告 `reports/vq2a8-opt3-20260908T235516715328Z/result/summary.txt` 的
`batched` 结果是 **TPOT 中位数 300.906 ms/token**、TTFT 中位数 1.035494 s。
配置为 Ascend950DT_9574、TP1、BF16 root、10 输入/4 输出、2 次 warmup、5 次正式测量。
“opt3” 是 V1 路径的优化实验名称，不是现在的 AscendC V3。
此处保留历史摘要数值，未获得该次原始 JSON；不承诺在新卡/v0.23 仍为 0.3 秒，
也不将它解释为 HTTP、长输出或全热缓存性能。

本次结束查看输出 `REPORT=...` 对应的 `result/summary.txt`，确认 `OPTIMIZATION_STATUS=PASS`，
再看 **batched** 的 TPOT（不要误读 baseline）。TPOT 按首 token 后到末 token 的时间除以
`输出 token 数 - 1` 计算，不包括模型加载时间，不是整次生成耗时除以 4。
短测只有 3 个 decode 间隔，不能代表长输出稳定性能。

本地 Windows 无 Ascend NPU；文档命令的计划检查和 CPU 回归不等于本次硬件复现完成。

V1 复现文档首次加入时：`--plan-only` 和相关 317 项 CPU 回归通过。完整 CPU 套件为
2417 passed、257 skipped、3 failed；其中 2 项为已有 FP8/FMA 舍入测试失败，
另 1 项为 Windows loopback 连接中止，单独复跑通过，未修改测试或放宽断言。
`bash format.sh ci` 因本地缺少 `pre-commit` 未能运行；`git diff --check` 通过。

环境检查修复验证（2026-09-14）：新增 89 项 CPU 用例，相关 484 项回归通过；完整 CPU 套件为
2507 passed、257 skipped、2 failed、4 subtests passed。两项失败仍是上述已知 FP8/FMA 用例，
未放宽数值要求。四个改动/新增 Python 文件 Ruff check/format 与 `git diff --check` 通过；
`bash format.sh ci` 仍因缺少 `pre-commit` 未运行完整 hooks。本次未在 NPU 上重编译或执行模型。

版本差异统一处理验证（2026-09-14）：本轮新增 95 项 CPU 用例，相关 579 项回归通过；完整 CPU 套件为
2602 passed、257 skipped、2 failed、4 subtests passed。失败仍为上述两项 FP8/FMA 用例，未修改它们或放宽断言。
覆盖同一规则下的元数据、子进程、逐行 pip 冲突分类、release 环境入口，以及成功警告和失败摘要。
六个改动/新增 Python 文件的 Ruff check/format、原复测命令的 `--plan-only` 和 `git diff --check` 通过；
`bash format.sh ci` 已尝试，因缺少 `pre-commit` 未完成。未改官方依赖安装 pins，未在 NPU 上执行或测量 TPOT。

移除自动一致性门槛验证（2026-09-14）：原 CPU 套件复跑为 2602 passed、257 skipped、2 failed、
4 subtests passed，失败仍是上述 FP8/FMA 用例。随后加入的 24 项默认 runtime-only 用例与相关回归共
603 项通过，覆盖默认不执行 pip/版本/源码审计，以及真实导入、设备、库和 SoC 失败仍会阻断。
12 个改动/新增 Python 文件 Ruff check/format 通过；`--metadata-only` 实际只记录，默认 CLI 在本机
实际报告缺少 `torch_npu`，不再报告版本不匹配。原复测命令 `--plan-only`、`git diff --check` 通过。
`bash format.sh ci` 仍因缺少 `pre-commit` 未完成；没有重编译或 NPU 实测。
