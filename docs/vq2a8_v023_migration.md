# VQ2A8 迁移到官方 vLLM Ascend 0.23.0

## 范围与验证边界

迁移分支为 `vllm-ascend-vq2a8-v023`，基于官方 `vllm-ascend v0.23.0`
提交 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`，配套官方 `vllm v0.23.0`。
原 `ascend950-vq2a8-v026` 分支、工作目录、虚拟环境和编译产物保留，不在原环境上降级安装。
分支推送、服务器安装、硬件验证是三个独立步骤。

本分支承接当前 0.26 分支新增的 VQ2A8 功能，而不是回到早期 TP1 快照：

迁移源快照为 `ascend950-vq2a8-v026` 的 `8b19204913b68d38bfdd06df457b6d230cec2220`。

- opt-in 模型注册、offline loader、root 权重处理和 QLI/SAS metadata 修正；
- 原 direct artifact、V1/V2/V3 原生实现和 V3 resident 执行路径；
- TP1/TP2 packed-zN 离线 repack、直接加载、算子形状检查与 TP2 通信；
- eager/fused preparation、TP1 MoE 图实验、标准 `vllm serve` 启动和 HTTP 流式测速；
- 显存预算、进度输出、通信诊断以及相应 CPU/设备验证工具。

框架适配以官方 0.23 的模型、调度器和通信接口为准，不整包覆盖成 0.26 的框架文件。
0.26 专属的 structured-output reasoning-boundary 补丁不适用于本分支。
TP2 设备映射使用 0.23 的 `device_id_to_physical_device_id`；不调用 0.26 才有的
`visible_device_id_to_physical_device_id`。本地版本后缀不再干扰 0.23 补丁选择。
默认 upstream 模型不会自动切换成 VQ2A8；TP2 仍只允许 `--decode-graph none`。

迁移代码、CPU 测试或环境检查均不等于新的 NPU 执行、模型质量、通信或性能 PASS。
本说明不宣称已在 0.23 环境完成原生编译、全模型推理或取得 TPOT 结果。

需要先回到原 V1 路径时，使用[V1 编译与复测命令](vq2a8_v1_reproduce.md)：
读取旧 direct 权重、按预算懒加载专家，并复测历史 `batched` 短请求 TPOT。
这不回滚 v0.23 框架或删除现有 zN repack 产物。
若继续排查 V3，保留[无模型权重启动诊断](vq2a8_tp1_startup_diagnose.md)入口。

## 配套环境

以下是官方 Ascend 0.23 与 vLLM 0.23 依赖的交集，不沿用 0.26 的 Transformers/FastAPI 约束。
这是新环境安装参考，不再作为现有环境运行验收时的精确版本门槛。

| 组件 | 要求 |
| --- | --- |
| 操作系统 / Python | Linux；Python `>=3.10,<3.13`，官方 A5 镜像使用 3.12 |
| vLLM | `==0.23.0`，从官方 tag 安装 `empty` 后端；允许本地版本后缀 |
| vLLM Ascend | 本迁移分支；可使用 editable 安装生成的 SCM 开发版本 |
| torch / torch-npu | `2.10.0` / `2.10.0.post4` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` |
| Transformers | `5.5.4` |
| Triton Ascend | `3.2.2`，不能与通用 CUDA `triton` 混装 |
| FastAPI | `>=0.115.0,<0.124.0` |
| Pydantic | `>=2.12.0`，官方未精确锁定小版本 |
| CANN / NNAL | 均为 `9.1.0`；950 使用对应平台包和兼容的驱动/固件 |

依据：[Ascend requirements](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/requirements.txt)、
[vLLM common requirements](https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/common.txt)、
[官方安装说明](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/docs/source/installation.md)、
[A5 镜像构建](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0/Dockerfile.a5)。
Ascend 的 `requirements.txt` 和 `pyproject.toml` 保留官方安装依赖，本次不修改或重装镜像里的包。

## 新目录安装：保留原 0.26 环境

以下命令供 Linux NPU 服务器执行，要求迁移分支已发布到 fork。
使用新终端，不要激活旧 0.26 venv，也不要修改正在运行的服务。
先确认本机已安装上述 CANN/NNAL；环境脚本路径按实际安装位置调整。
示例 Python 3.11 在支持范围内；也可替换成独立的 Python 3.12 解释器。

```bash
set -e
VQ2_V023_ROOT=/home/g00872988/vllm-ascend-vq2a8-v023

# 新路径已存在时停止；不要删除或覆盖现有 checkout。
test ! -e "$VQ2_V023_ROOT"
git clone --branch vllm-ascend-vq2a8-v023 \
  https://github.com/YChange01/vllm-ascend-vq2a8.git "$VQ2_V023_ROOT"
cd "$VQ2_V023_ROOT"
git merge-base --is-ancestor 5cb98caaadeff42b5b62b996e34bb2aaa29d20fd HEAD

unset PYTHONPATH VLLM_VERSION VLLM_VERSION_OVERRIDE VLLM_TARGET_DEVICE
unset SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM
unset SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_ASCEND VLLM_USE_PRECOMPILED VLLM_USE_PRECOMPILED_RUST
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
/usr/local/python3.11.10/bin/python3 -m venv .venv-v023
source .venv-v023/bin/activate
python -m pip install --upgrade pip

# 先准备 Ascend 的运行时依赖和同环境构建工具。
python -m pip install -r requirements.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install 'cmake>=3.26.1' ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<81' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9' wheel jinja2

# 新目录内单独保存官方 vLLM 源码；不要使用通用 CUDA/CPU 安装入口。
git clone --branch v0.23.0 --depth 1 \
  https://github.com/vllm-project/vllm.git .upstream-vllm-v023
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation \
  -c requirements.txt -e .upstream-vllm-v023

# 编译并 editable 安装本分支的 Ascend 插件和自定义算子。
git submodule update --init --recursive
env -u SOC_VERSION \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0+vq2a8.v023 \
  COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=4 \
  python -m pip install --no-build-isolation -e . \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://download.pytorch.org/whl/cpu

python -u tools/validate_vq2a8_v023_environment.py
```

`empty` 表示上游 vLLM 不编译 CUDA/CPU 扩展，由 Ascend 插件提供 NPU 后端，并非使用 CPU 推理。
官方 vLLM 的 CPU requirements 和隔离构建环境会引入 torch 2.11；这里按 Ascend 环境预装依赖，
再用 `--no-build-isolation` 安装 `empty` 源码，避免与 torch-npu 2.10 的要求混用。
CPU wheel 索引用于获取与 torch-npu 配套的 PyTorch，不等于 `VLLM_TARGET_DEVICE=cpu`。

主插件构建默认通过 `npu-smi` 检测 SOC；若无法识别，先确认本机 CANN 支持的芯片型号，
将 `env -u SOC_VERSION` 改为显式的、匹配硬件的 `SOC_VERSION=...`。
不要直接照抄官方镜像的 `ascend950dt_9582` 或其他机器的后缀。

`setup.py` 使用 `setuptools_scm.get_version()`。迁移分支在 release tag 后增加提交时，
可自动生成 `0.23.1.dev5+g32c3714e4` 一类版本号，**不代表装成了另一个框架分支**。
已经完成 editable 安装和编译时，不必为了版本号重装或重编译。

### 自动运行不做一致性审计

按用户要求，所有 VQ2 自动验收、benchmark、offline 和 demo 子进程不再强制比较 Python 包版本、
检查 editable/Git 源码来源或运行全局 `pip check`。不增加开发版本白名单，也不要求额外的跳过参数。
报告只记录实际版本，并明确标记 `validation_profile=runtime_only`、`consistency_checked=false`、
`pip_check_run=false`；不会把未执行的检查写成通过。

验收的 `environment` 阶段仅保留模型实际使用的导入和接口检查；后续设备占用、算子库/SoC、
权重格式与数值预检仍照常执行。`affinity-sched`、`ms-service-profiler` 等包的全局依赖报告不会阻断复测；
若真实导入或算子执行缺少依赖，仍会显示实际异常并停止。移除一致性门槛不等于证明所有环境都兼容。

```bash
python -c 'import sys, vllm, vllm_ascend; print(sys.executable); print(vllm.__file__); print(vllm_ascend.__file__)'
python tools/validate_vq2a8_v023_environment.py --metadata-only
```

三个路径应分别落在新 venv、新 `.upstream-vllm-v023` 和新 Ascend checkout 中。
`--metadata-only` 仅记录发行包信息，返回 `status=recorded`；默认模式执行真实导入和调度接口检查，
不加载模型权重或验证设备算子。原有严格诊断仅供手动排查：

```bash
python tools/validate_vq2a8_v023_environment.py --audit-consistency
```

只有显式使用这个选项才执行旧版本/来源规则和全局 `pip check`，它不是复测前置条件。
旧报告复用（`--resume`）和旧 baseline 对比仍要求证据对应当前环境；环境变更时使用新报告重跑。

## CPU 回归与原生库重建

在新 venv 安装 pytest 后可运行隔离 CPU contract suite：

```bash
python -m pip install pytest
python -X utf8 tools/run_vq2a8_cpu_tests.py
```

该入口使用真实 CPU PyTorch 做张量断言，隔离 vLLM/torch-npu 初始化，并拒绝 Triton kernel launch。
它不是 vLLM 引擎、NPU 模拟器或硬件验收；保留输出中的失败与 skip，不自动放宽精度门槛。

新的工作目录具有独立 `build/`。即使 packed 格式未变，也不要复制或软链旧 0.26 的 `.so`、
CMake cache 或 build manifest。主插件安装完成后，V3 服务还需要单独构建 V3 库：

```bash
# 示例来自 Ascend950DT_9574；仅在与所选设备/工具链匹配时使用。
python -u tools/build_vq2a8_ascendc_v3.py \
  --soc Ascend950DT_9574 \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --jobs 4 --timeout 1800
```

默认输出为新目录下 `build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so`。
若选用 V1/V2 执行策略，则分别使用 `tools/build_vq2a8_ascendc.py` 或
`tools/build_vq2a8_ascendc_v2.py` 的对应构建参数；V3 服务无需先把三代库全部编译。
`BUILD=PASS` 只代表构建成功，不代表 `DEVICE_EXECUTION_VERIFIED=True`。

## 复用 artifact，启动 TP1 或 TP2

现有完整 TP1 `vq2a8_zn_tp1_v1` 和 TP2 `vq2a8_zn_tp2_v1` artifact 沿用同一 packed-zN/K256 pair-LUT 契约。
**不需要仅因框架从 0.26 迁移到 0.23 重新 repack 或复制模型。**
保留 manifest、SHA、完整性、形状和 runtime 校验；不要手改验证标记来绕过拒绝。
旧 direct artifact 仍走其原有路径，不能仅通过目录改名把 direct 格式当作 zN。

先确认设备空闲且获得使用许可。下面两种服务启动方式二选一；端口冲突时使用新端口，
不要停止其他人的服务。TP1 repack 只减少启动布局转换，不减少全驻留模型所需 HBM；
显存预算不通过时仍必须停止，不能因为迁移而取消安全预留。

TP1 示例：

```bash
python -u tools/serve_vq2a8_v3.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_tp1_zn \
  --tensor-parallel-size 1 --physical-npu 0 \
  --port 8000 --preparation eager --decode-graph none
```

TP2 示例：

```bash
python -u tools/serve_vq2a8_v3.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_tp2_zn \
  --tensor-parallel-size 2 --physical-npus 0,1 \
  --port 8000 --preparation eager --decode-graph none
```

参见 [TP1 zN 加载说明](vq2a8_tp1_zn_offline.md) 和
[TP2 算子/通信检查](vq2a8_tp2_offline.md)。这些文档中的原仓库 `cd` 路径在迁移时改为新目录，
本迁移说明的“新环境重新编译”要求优先于旧文档针对一次 Python-only 更新的“无需重编译”说明。
TP2 应先用 `tools/validate_vq2a8_tp2_collective.py --communication-only` 的双进程流程检查 HCCL，
而不是反复加载完整模型；框架迁移不保证修复底层 UB/HCCL 连接故障。

服务就绪后，另一个终端使用新 venv 测量；服务保持运行可反复执行，无需重新加载：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
source .venv-v023/bin/activate
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 \
  --max-tokens 32 --warmups 1 --repeats 3
```

输出是客户端 HTTP 流式 TTFT/TPOT，不是纯 kernel 时间。保留本次环境、库 SHA、参数和实际结果；
不得将旧环境的验收报告或 TPOT 当作新环境的结果，也不应为跨版本复用旧报告而删除版本校验。
若需回退，退出新服务后重新使用原 0.26 checkout/venv 和原库；无需在新 venv 里反向覆盖安装。

## 本次本地验证记录（2026-09-14）

验证机为 Windows / Python 3.14.7 / CPU PyTorch 2.10.0，仅用于 CPU contracts，
不是上述受支持的 NPU 运行环境。

- 完整 CPU suite 首轮：`2271 passed, 2 failed, 257 skipped, 4 subtests passed`。
  两项失败在迁移源 `8b1920491` 上用同一解释器、`-X utf8` 精确复测后也存在：
  `test_inverse_rope_fp8_midpoint_regression_and_bounded_diagnostics` 和
  `test_rounding_probe_requires_exact_inputs_scales_and_fp32_before_weight_gates`
  （均在 `test_vq2a8_root_fp8.py`，涉及本机 FP8/FMA 舍入位模式）。未加 xfail 或放宽数值断言。
- 主代理完整复跑：`2270 passed, 3 failed, 257 skipped, 4 subtests passed`。
  额外一次失败为 `test_real_loopback_http_routes_validation_unicode_and_failed_health` 的
  Windows 本地 HTTP `ConnectionAbortedError / WinError10053`。新旧分支各单独重测两次均通过，
  本轮未复现，根因未确认；不能据此把它认定为旧分支已有故障，也未添加重试掩盖失败。
- 迁移专项复核（`-k 'v023 or tp2_integration or v3_prepare_native'`）：
  `161 passed, 4 skipped, 2365 deselected`。
- `csrc/vq2a8_expert_reference/tests`：`63 passed, 49 subtests passed`。
- 全部 207 个原新增文件已迁移，版本化环境/测试入口重命名为 v023；50 个 VQ2 原生源码文件
  与迁移源在归一化文本换行后完全一致，packed-zN 格式与算子数学逻辑未改。
- 改动及新增 Python 文件的 Ruff check/format、AST 语法检查通过；四个 VQ2 Bash 工具语法检查通过。
- 官方 0.23 的 requirements、pyproject、scheduler、structured-output patch 和 model-runner 文件保持原样。
  针对本地版本后缀、TP2 设备映射、真实输入 ID、原始模型注册和上游核心文件保留新增回归检查。
- `bash format.sh ci` 已尝试，但本机缺少 `pre-commit`，未运行完整 hook 集合；不能据此宣称 CI 通过。

NPU 构建、TP1/TP2 模型执行、HCCL 和性能验收尚未在此环境完成。
