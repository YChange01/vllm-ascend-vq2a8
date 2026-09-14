# V4：V1 batched 算术路径，全量专家常驻 HBM

V4 是独立的 `execution_policy=ascendc_v4`。它复用 V1 的 native `.so`、
`vq2a8_direct_tp1_v1` 权重格式、rowwise activation preparation，以及 batched 路由和归约。
不调用 V3 kernel/packed-zN/graph，也不修改原 V1 benchmark 矩阵。
默认行为不变；新增可选 `--device-route-decode` 的构建、预检和对照流程见下文。

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

### 小显存短文本试跑：设备选择值 2，输入 10 / 输出 4 token

这是显式收紧配置的试跑方式，不改变上述默认值。`--physical-npu` 是沿用的参数名，
其值原样传给 `ASCEND_RT_VISIBLE_DEVICES`；容器 runtime 设备编号不一定等于 `npu-smi` 的 NPU ID。
先用 `npu-smi info -m` 核对映射并确认实际目标卡健康、空闲；不要使用或停止别人的卡和进程。
下面以已经确认的选择值 `2` 为例。未启用 device-route 时，已有 V1 `.so` 不需要重编。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
git pull --ff-only origin vllm-ascend-vq2a8-v023
python -u tools/serve_vq2a8_v4.py --model /home/g00872988/vq2a8 --library build/vq2a8-ascendc-v023-v1/libvq2a8_ascendc.so --physical-npu 2 --max-model-len 16 --kv-cache-mib 256 --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 --port 8000
```

`--max-model-len 16` 同时把 `max_num_batched_tokens` 设为 16，缩小启动 profile 的输入；
并不跳过 profile。`--kv-cache-mib 256` 将固定 KV 预算从 1 GiB 降为 256 MiB，节省 768 MiB。
block size 仍为后端要求的 128，不需要随上下文改为 16。总输入加输出最多 16 token，保持单并发。

`--memory-fraction 1.0` 只放宽专家常驻预算的比例上限，不改变 engine 的 0.9，
也不会把所有剩余显存填满或自动挤掉其他任务。保留 3 GiB 供 KV、激活、临时空间等使用；
这是预算余量，不是预先分配的张量。全专家必须完整放下，仍不允许缺专家或缓存回退。

以先前失败日志的同一份快照计算：已分配 roots 等约 14.78 GiB，全部专家约 61.54 GiB，
二者合计约 76.32 GiB，并不是整个模型只占 61.54 GiB。
原 0.9 / 8 GiB 配置只给专家约 49.36 GiB；改成 1.0 / 3 GiB 后理论预算约 61.67 GiB，
仅比专家计划多约 128 MiB。卡 2 实际余量须以新启动的 `MODEL_CACHE_BUDGET` 为准，
不能把旧卡快照当成保证。缩短文本不会减少权重大小；通过此预算仍可能在 profile 或请求期间 OOM。
若仍失败，保留第一条错误和预算日志，不移除检查或自动继续减小 reserve。

服务 ready 后，同一容器的另一终端可用以下请求验证精确的 10 个输入 token、4 个输出 token。
这里用模型自身 tokenizer 生成 token ID，不以字符数代替 token 数；`ignore_eos` 用于固定输出长度。
请求字段参考 [vLLM completions 协议](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/)。

```bash
python - <<'PY'
import json
from urllib.request import Request, urlopen
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('/home/g00872988/vq2a8/tokenizer.json')
ids = tokenizer.encode('Explain why the sky is blue. ' * 4, add_special_tokens=False).ids[:10]
assert len(ids) == 10
payload = {'model': 'vq2a8', 'prompt': ids, 'max_tokens': 4,
           'temperature': 0, 'ignore_eos': True, 'add_special_tokens': False}
request = Request('http://127.0.0.1:8000/v1/completions',
                  data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
with urlopen(request, timeout=300) as response:
    result = json.load(response)
print(result['choices'][0]['text'])
print('USAGE:', result['usage'])
assert result['usage']['prompt_tokens'] == 10
assert result['usage']['completion_tokens'] == 4
PY
```

此请求只检查服务和 token 数，不测稳定 TPOT；不能据此宣称 0.3 s/token。

## 可选优化：单 token device-route decode

这个开关只影响 V4 的单 token 路径：路由 ID 留在设备上，由 native kernel 选择常驻专家及准备描述符，
不做原来的逐层 `ids.cpu().tolist()` 或 singleton 描述符上传。多 token prefill 仍走原来的 batched 路径；
单 token prefill 也可使用 singleton 路径。rowwise RHT/量化舍入、V1 packed 权重、投影和 slot 混合顺序保持不变。
这不表示整个模型无 CPU 同步或所有 DMA 为零：采样、有效性检查和其余模型工作仍存在。

默认不打开这个开关，旧库仍可运行原 V4。启用它需要含 `ResidentBank` ABI 的新库，
缺符号会明确失败；不会静默退回 CPU 路由。直接权重 artifact 不变，不需要 repack。
先停止自己在目标卡上的服务，再在独立构建目录编译；保留旧库用于回退。
以下使用本机日志中的 `Ascend950DT_9582`；换机器时必须核对当前目标设备的准确 SoC，不能混用另一台机器的值。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/build_vq2a8_ascendc.py \
  --soc Ascend950DT_9582 --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-device-route --jobs 4
python -u tools/validate_vq2a8_v4_device_route.py \
  --library build/vq2a8-ascendc-v4-device-route/libvq2a8_ascendc.so \
  --physical-npu 2 --report-dir reports/v4-device-route-preflight
```

预检目录必须是尚不存在的新目录；重复运行时换一个目录名。它只运行小型合成数据，
不加载完整模型或专家文件。必须看到 `summary.json` 中 `status=PASS` 和
`device_execution_verified=true`；这只验证合成 native 路径，不代表全模型数值或速度通过。

### 修复混合投影路径的回调释放与队列等锁

第一版只将 `ResidentBank::Select/Project` 改为 `OpCommand::RunOpApi(..., false)`，
目标设备的 `--queue-lifetime` 回归仍出现超时，另一轮出现 allocator `try_merge_blocks` 内部断言。
后续双线程 native 栈直接显示：主线程在 `eq_Scalar` 入队过程中析构旧 `Run` 回调，
释放线程则在析构新的 `ResidentBank::Project` 回调，并经 `NPUEvent::record` 再次入队。
结合源码，这高度符合入队锁与分配器锁的反向等待；具体 mutex owner 未直接读取。
不能将停留在 `eq` / `matmul` 的 Python 行号解释成该计算本身耗时过长。

本次补齐同一个 `.so` 中 `torch_binding.cpp` 的 `Run` 和 `GroupedProjection`（包括 Pipeline 模板），
统一改用 `OpCommand::RunOpApi(..., false)`，避免遗留 `SetCustomHandler/Run` 路径
在队列槽复用的临界区析构 Tensor-owning handler。
保留全部 Tensor/state/vector/descriptor 捕获、ResidentBank 的 `recordStream`、同流限制及 V1 算术；
不删除生命周期保护，不新增逐层同步，不切换图模式。
回调转移行为可参照 [torch-npu OPAPI 释放实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/framework/OpParamMaker.cpp#L806)。
这是针对已观察到的混合路径的候选修复；尚需目标机器验证，也不能据此断言另一轮 allocator 崩溃已解决。

先保留现场日志，停止自己原来的服务，确认目标卡空闲；不要停止其他任务或重置整卡。
同步本次源码后在独立目录重编，只构建专用 `.so`，无需重新安装 vllm-ascend、编译全部 OPP 或 repack 权重。
以下使用本次确认的卡 1；旧 V4 基线库保持不动。

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/build_vq2a8_ascendc.py \
  --soc Ascend950DT_9582 --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-device-route-fix2 --jobs 4
python -u tools/validate_vq2a8_v4_device_route.py \
  --library build/vq2a8-ascendc-v4-device-route-fix2/libvq2a8_ascendc.so \
  --physical-npu 1 --queue-lifetime --timeout-s 300
```

`--queue-lifetime` 是可选的小型合成回归，不读模型或专家文件；保留原短预检，
在同一个子进程中追加连续提交、临时 Tensor 丢引用、队列槽复用和混合投影压力测试。
不能以拆分后的独立 ResidentBank 测试通过代替混合回归通过。压力分两个阶段：

- `queue_lifetime_wrap`：保留完整的 ResidentBank select/project 与普通 matmul 压力。
- `queue_lifetime_mixed`：再交替调用普通投影、分组投影、Pipeline 分组投影、ResidentBank 和普通算子。

默认每阶段 2049 轮，仍使用小型合成专家。第一阶段不引入 grouped 的描述符上传；
第二阶段保留 grouped 原有的阻塞式描述符拷贝，并在报告中注明，不能将它称为零同步压力。
两阶段共至少 18441 次相关算子调用；这是源码调用下界，不是实际队列槽位测量。
保持 task queue 默认启用及 `--launch-blocking 0`；
不要用关闭队列或同步启动的方式掩盖问题。循环不显式逐轮同步，仅在测试边界检查结果；
原生同流检查仍可能等待 host 队列，因此这不是“无 CPU 等待”或性能测试，也不声称直接测得队列索引。
默认自动创建新的 `/tmp/vq2-v4-device-route-*` 报告目录。超时/失败都不代表通过，不要继续启动全模型或反复压测。

`QUEUE_STEP` 只在前 8 轮及每 256 轮打印阶段、轮数、步骤和 `BEGIN/RETURN/FAIL`；
拆开了输入 clone、native 调用、普通 matmul、结果校验及丢引用的位置。
它们是纯 host 标记，`RETURN` 不表示设备已经完成，不新增同步或读取 Tensor 值。
显式设备同步边界为 setup、原压力阶段结束、混合阶段结束；三者及最终正确性检查必须全部通过。

回归通过并释放设备后，用全新进程启动原服务命令，只将 `--library` 换成上述 `-fix2` 目录的库，
保留 `--device-route-decode`。已加载旧库的进程不能通过重编或再次 `load_library` 热修复。
先确认一次 1/4 及 10/4 请求能够返回，再按原参数测 TTFT/TPOT；本修复不承诺固定加速比。

### 同引擎对照与服务测速

以下命令沿用原构建目录；若验证上述修复，将各处 `--library` 统一换为 `-fix2` 目录的新库。

先做同一常驻模型内的数值和计时对照（此时不要同时启动 HTTP 服务）：

```bash
python -u tools/accept_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-device-route/libvq2a8_ascendc.so \
  --physical-npu 2 --device-route-decode --cases 10:4 \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --cache-reserve-gib 3
```

不加 `--compare-v1`，所以只加载一次 V4 完整模型。先比较同引擎 batched 与 candidate 的 tokens/logits
bit-exact，再检查 candidate 重复一致性；不一致就停止，不放宽容差。两条路径各预热 2 次后，
按 AB/BA 交替各实测 5 次，切换模式不重载权重或清空常驻缓存。对照双方均保留相同的常驻 metadata banks。
启用此开关的同引擎 A/B 双方在每次 model 输出前都执行一次合并有效性 `bool` 检查，其开销包含在各自计时中；
不是只给 candidate 额外逐层同步。未启用开关的原服务路径不改变。
`V4_RESULT` 的 `tpot_median_s` 是 candidate，`same_engine_batched_tpot_median_s` 是 baseline；
`tpot_ratio_vs_same_engine_batched` 小于 1 才表示本轮 candidate 更快，不预设加速比例。
`--max-model-len`、`--kv-cache-mib` 和两项 memory fraction 同样传给 benchmark worker；
所有 `--cases` 必须在加载前满足输入加输出不超过上下文上限。旧默认值仍是 128 / 1024 MiB / engine 0.9。

验收退出、目标卡资源释放后，终端 1 启动 candidate 服务：

```bash
python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-device-route/libvq2a8_ascendc.so \
  --physical-npu 2 --device-route-decode \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 --port 8000
```

服务就绪后，终端 2 用相同 10/4 长度和单并发测速（不要再启动另一份完整模型）：

```bash
python -m vllm.entrypoints.cli.main bench serve \
  --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --model vq2a8 --tokenizer /home/g00872988/vq2a8 --dataset-name random \
  --random-input-len 10 --random-output-len 4 --random-range-ratio 1 \
  --num-prompts 5 --num-warmups 2 --max-concurrency 1 --request-rate inf --ignore-eos
```

核对输出中的实际输入/输出 token 总数；目标是 5 个计时请求共 50 / 20 token。
HTTP 的 TTFT/TPOT 包含客户端、服务调度、IPC 和输出处理，不能直接当作离线 engine-step 计时。
未提供运行时 HTTP 热切换：回退服务时停止自己的进程，去掉 `--device-route-decode`，再按原配置启动。

candidate 每个请求都用前后计数差值确认 43 层实际使用新路径，而不是用预热留下的累计值充当覆盖率。
对于 10/4，每层本次应有 `singleton_forwards=3`、`batched_prefill_forwards=1`、`device_select_calls=6`，
以及 singleton 路由 host reads、描述符 H2D 为零。这里是运行路径计数，报告明确
`runtime_path_counters_not_profiler_measured_DMA`；仍需 profiler 才能声称实际总传输量或 kernel 耗时。
新路径的 NPU 数值、显存峰值和性能尚需在目标机器上按上述步骤实测；CPU 测试不能替代这些结果。

## 默认验收：只加载一次 V4

先结束自己在目标卡上的旧服务并确认目标卡空闲，不停止其他人的任务。
在普通硬件 CANN 环境下运行，不能带 simulator 配置。这里复用已编译的 V1 库，不需要重新 repack。
已有下列 V1 库及对应构建记录时无需重编；尚未构建时，按 [V1 编译指南](vq2a8_v1_reproduce.md)
用当前卡的准确 `--soc` 构建到 `build/vq2a8-ascendc-v023-v1`。默认路径没有单独的 V4 `.so`；
启用上面的 device-route 才需要重建含新 ABI 的库。

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
`ids.cpu().tolist()` 和其他模型工作在默认 batched 路径仍存在；device-route 的 singleton 差异见上节。

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

短文本 CLI 补充：参数测试 49 项通过；全量 CPU 回归为 2907 passed、257 skipped、2 failed，
仍为上述两个原有 FP8/FMA 用例。Ruff 通过，完整 hooks 仍因缺少 pre-commit 未能完成。
16 token / 256 MiB 配置仅完成静态与 CPU 检查，未在物理卡 2 实测显存峰值或启动成功。

Device-route 补充：V4/router/optimization 定向 CPU 回归为 575 passed、1 skipped；
全量为 3016 passed、258 skipped、2 failed（仍为上述既有 FP8 midpoint / CPU FMA 用例），另有 4 个 subtests 通过。
新增原生边界测试因缺少主机 C++ 编译器而跳过；Python Ruff 检查、格式检查及新增 C++ 文件 clang-format 通过。
完整 `format.sh ci` 仍因本地没有 pre-commit 未能执行。尚未完成 CANN 编译、NPU 精确数值或性能实测；
不将源码检查和 CPU oracle 通过解释成设备通过，也不据此声称 TPOT 已下降。

device-route 入口补充：V4 服务 CLI、验收编排、AB/BA 调度、计数差值和回执校验共 162 项 CPU 测试通过，
对应 Python 文件 Ruff 检查与格式化通过。包含默认关闭、预算透传、上下文超限、缺少对照证据和短预检失败即停止等用例；
该结果不包含 native 编译、目标 NPU 数值或加速验证。

回调释放锁重入修复：新增 2 项 native 源码契约回归（修复前失败、修复后通过）和 20 项小测试编排用例。
V4/activation/optimization 定向 CPU 回归为 667 passed、1 skipped；最终全量为
3038 passed、258 skipped、2 failed，另有 4 个 subtests 通过。两项失败仍为上述既有 FP8 midpoint / CPU FMA 用例。
Ruff、clang-format 和 `git diff --check` 通过；`format.sh ci` 仍受本地缺少 pre-commit 限制。
本地没有 CANN/NPU，尚未编译新 `.so` 或运行实机异步回归，不能据 CPU 测试宣称服务死锁已在设备上消除。

混合回调修复（第二版）：新增 V1 提交入口源码契约，旧代码为 3 failed / 1 passed，
补齐两处封装后通过；覆盖原 Tensor 捕获、Launch 实参及 Pipeline 模板入口不变。
混合生命周期与 native 契约定向 CPU 回归为 61 passed / 1 skipped，跳过项缺少主机 C++ 编译器。
最终全量 CPU 回归为 3050 passed / 258 skipped / 2 failed，另有 4 个 subtests 通过；
两项失败仍是上述既有 FP8 midpoint / CPU FMA 用例，本次未修改该路径。
弱引用测试确认临时输入/输出按预期释放，原 ResidentBank 压力阶段没有插入 grouped 描述符拷贝，
单投影、grouped、pipeline 各路径的错误会使回归失败。Ruff、clang-format 和 `git diff --check` 通过；
完整 `format.sh ci` 仍因本地缺少 pre-commit 未执行完成。此结果不是 CANN 编译、NPU 数值或死锁消除证明。
