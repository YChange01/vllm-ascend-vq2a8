# V4：V1 batched 算术路径，全量专家常驻 HBM

V4 是独立的 `execution_policy=ascendc_v4`。它复用 V1 的 native `.so`、
`vq2a8_direct_tp1_v1` 权重格式、rowwise activation preparation，以及 batched 路由和归约。
不调用 V3 kernel/packed-zN/graph，也不修改原 V1 benchmark 矩阵。

完整根权重严格加载成功后，V4 先核对全部专家预算，再预加载全部专家。
预算不足、加载不完整、运行中缺少专家都报错；不会退回懒加载或自动降低 reserve。
首次 worker dummy/profile 保留 V1 的原始几何；`LLM` 初始化返回后才按 V1 的时点配置 batched。

## 直接启动 vLLM HTTP 服务

这是独立于下面验收脚本的快捷入口：不编译、不跑验收矩阵、不检查包版本或全局依赖，
直接启动标准 `vllm serve`，模型和全部专家只加载一次，服务保持运行。
先确认物理卡 1 空闲；已有 V1 `.so` 可直接复用，无需重新编译或 repack。

终端 1（容器内）：

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023
python -u tools/serve_vq2a8_v4.py --model /home/g00872988/vq2a8 --library build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so --physical-npu 1 --reserve-gib 8 --port 8000
```

等到 `Application startup complete`，在同一容器的终端 2 请求（不要同时运行验收脚本）：

```bash
curl http://127.0.0.1:8000/v1/models
curl -N http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"vq2a8","prompt":"请用一句话介绍你自己。","max_tokens":32,"temperature":0,"stream":true}'
```

这里使用 `/v1/completions`，不依赖尚未确认的 chat template。输入和输出 token 总数不超过 128，
单并发、TP1、BF16、eager；不是长上下文或生产并发配置。默认仅监听 `127.0.0.1`。
其他容器或远程机器不能直接使用这个 loopback 地址；跨主机开放监听前需自行配置认证和访问控制。
`--dry-run` 仅打印命令，不初始化 NPU。正常停止服务在终端 1 按 Ctrl+C。

服务模式在严格权重加载、V4 全专家常驻完成后关闭离线逐层诊断同步/日志。
启动 profile 仍保留 V1 原始 `token_chunk=2`；首个带真实 attention metadata 的 forward
先同步一次、确认常驻完整，再启用 V1 `batched`，终端打印 `MODEL_V4_SERVING_READY`。
后续请求不重复切换、不扫描全部专家元数据、不清空缓存，也不积累离线 logits/trace。
首次请求包括一次性切换和常量准备开销，不用它声称稳定 TPOT。
HTTP 返回成功并不等同于通过下面的数值验收或性能测试。

## 默认验收：只加载一次 V4

先结束自己在目标卡上的旧服务并确认目标卡空闲，不停止其他人的任务。
在普通硬件 CANN 环境下运行，不能带 simulator 配置。这里复用已编译的 V1 库，不需要重新 repack。
已有下列 V1 库及对应构建记录时无需重编；尚未构建时，按 [V1 编译指南](vq2a8_v1_reproduce.md)
用当前卡的准确 `--soc` 构建到 `build/vq2a8-ascendc-v023-v1`。没有新的 V4 `.so`。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023
python -u tools/accept_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so \
  --physical-npu 1 \
  --cache-reserve-gib 8
```

本入口固定使用模型目录下的 `experts_vq_ascend_v2`。目录名带 v2，但格式仍是 V1 的
`vq2a8_direct_tp1_v1`；不是 TP2/TP1 packed-zN。本轮不提供自定义 artifact 参数，
确保原生短预检和完整模型使用同一个标准 V1 artifact 目录。

默认顺序是 runtime imports/API 检查、原生短预检、独立 V4 worker。
没有版本 pin、editable 来源或全局 `pip check` 一致性门槛；实际导入、设备、SoC、库身份、
权重和数值错误仍停止验收。默认每个子进程超时为 14400 秒，可用 `--timeout` 修改。
默认选择物理卡 1。每个设备阶段前，以及完整模型 worker 初始化设备前，都会只读检查所选卡是否空闲；
占用、缺卡、查询失败或无法识别均停止，不自动结束别人的任务。其他卡的占用不会阻止所选卡测试。
这是运行前快照，不是独占预约；仍需要与其他使用者协调。前一子进程退出但资源尚未释放时也会停止。

默认 reserve 为 8 GiB，可显式调节，不会自动降低。该余量还需覆盖 KV、attention、
activation 临时张量和分配器开销；通过静态预算不等于能保证整个请求不 OOM。
当前典型 43 层 direct 模型（前三层各 1 专家，后 40 层各 256 专家）的全专家逻辑 payload 约
61.54 GiB；实际计划按 artifact header 计算，不硬编码该数值。这不含 roots、KV 和激活，
设备显示 98G 不等于全部可分配给专家；以本次 `MODEL_CACHE_BUDGET` 和 `MODEL_CACHE_PLAN` 为准。
确认命令而不接触设备可以加 `--plan-only`。

## 可选：与 V1 batched 做实测对照

```bash
python -u tools/accept_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so \
  --physical-npu 1 \
  --cache-reserve-gib 8 \
  --compare-v1
```

这会先运行一个 V1 batched 子进程，保存诊断 logits/tokens 和测速结果；进程结束后，
再加载 V4。不会同时保留两个完整模型。V4 要求相同输入、库、源码和设备身份，并逐 case
检查 tokens 和 logits bit-exact；不一致就停止该次验收，不放宽容差。

未加 `--compare-v1` 时，报告明确 `v1_comparison=NOT_RUN`：只验证 V4 的真实执行、
有限值、重复一致性及常驻约束，不宣称已经实测证明 V1/V4 等价或模型质量正确。

## 报告与计时口径

终端打印 `V4_STAGE`、专家预加载进度、`V4_ENGINE_READY`、`V4_SAMPLE`、`V4_RESULT`、`V4_ACCEPTANCE`
和报告目录；`V4_RESULT` 列出各 case 的实测 TTFT/TPOT/E2E 中位数及专家搬运计数。默认报告在
`reports/vq2a8-v4-*`，也可指定一个不存在的新 `--output-dir`。

- `run.json`：各子进程退出码、超时、日志路径与总状态。
- `v4/summary.json`：实际请求的 TTFT/TPOT/E2E、逐 token 交付时间、NPU event interval、
  native calls/launches、显存、cache 与专家 payload H2D 计数。
- `startup_snapshot.v4` 和 `preload`：各层预加载耗时、专家数量及实际 payload H2D 成本。
  `engine_init_profile_kv_s` 包含模型加载、专家预加载、worker profile 和 KV 初始化，
  不作为稳定请求 TPOT。
- 每个 case 保存两次诊断 logits 和证据；可选 V1 报告在 `v1_reference/`。

默认先测试小用例 `10:4`（10 个输入 token、4 个输出 token），先进行两次非计时诊断，再预热 2 次、
实测 5 次。扩展测试可显式传 `--cases 10:4,32:32,96:32`。
可以用 `--cases`、`--warmups`、`--repeats` 修改，但上下文总长不超过 128，
至少 2 次预热和 5 次实测。仅支持 TP1、B1、eager、BF16 roots。

每个 V4 请求必须保持 `cache_delta.loads=0`、`cache_delta.evictions=0`、
`expert_payload_h2d_bytes=0`，并通过所有层常驻完整性校验。
这些是“专家 payload 无请求期搬运”，不是所有 DMA/H2D 为零：路由索引上传、
`ids.cpu().tolist()` 和其他模型工作仍存在。

计时复用 V1 的真实 engine step 与边界同步。NPU event interval 包含 host 提交空隙，
phase timer 是 host submission/wait 时间，均不是纯 kernel 耗时。
V1 热缓存本来就可能无专家搬运，因此 V4 不保证固定速度提升。
这些结果不是 HTTP/并发性能、长期稳定性、模型质量或 full-model graph 认证。

## 本次开发验证

在 Windows CPU 环境中，V4 及相关缓存、offline、execution 回归共 561 项通过；
全量 CPU 测试为 2830 passed、257 skipped、2 failed。两个失败均为原有
`test_vq2a8_root_fp8.py` 的 FP8 midpoint / CPU FMA 数值用例，本次没有修改该实现。
改动的 Python 文件通过 Ruff 检查与格式检查；`format.sh ci` 因本地缺少 pre-commit 未能完成。
尚未进行 NPU 实测，不能据此认定 V4 已达到 0.3 s TPOT，也不能认定已解决此前 V3 的阻塞。

HTTP 入口补充验证：新增 58 项 CPU 测试，最终相关回归 493 项通过。全量复跑时为
2885 passed、257 skipped、2 failed，之后补充的三个 CLI/配置用例已包含在最终相关回归中；
两个失败仍是上述原有 FP8/FMA 用例。服务 Python 改动通过 Ruff 检查和格式检查，
完整 hooks 仍受本地缺少 pre-commit 限制；尚未在 NPU 上启动或通过 HTTP 请求实测。
