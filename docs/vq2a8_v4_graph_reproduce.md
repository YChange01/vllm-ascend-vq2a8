# V4 MoE decode graph：验证、服务与回滚

本轮实现显式 `--decode-graph moe` 接入，不把 CPU 单元测试当作 NPU 验收。
当前交付环境没有执行新图的 Ascend 硬件验证，也没有新图的 TTFT/TPOT 实测结果。
既有 fix2 eager 小测试 PASS 与 HTTP 时延下降，不等于本图路径已通过硬件验收。

本地验证：1054 项 CPU 测试通过、5 项 Linux/主机 C++ 编译器相关测试跳过；
覆盖 V4 图/服务/验收协议，以及原 activation、execution、offline、V3 和 AscendC 回归。
测试使用真实 CPU Tensor 数值运算和显式 fake graph backend；隔离了不可用的
vLLM 平台插件，未调用的 accelerator 导入替身一旦执行即报错，不模拟 NPU 成功。
20 个变更 Python 文件通过 Ruff 检查/格式检查与 Python 3.11 语法解析，另通过 diff 检查。
这些结果不证明目标 torch-npu/CANN 能捕获或加速自定义 native 算子；以下实机探针仍是必需步骤。

范围固定为 V4、TP1、B1、BF16 roots、原 V1 packed 投影数学。
MoE 图包含设备路由、稀疏 lookup、gate/up、原 rowwise preparation、SwiGLU、down、
有序 FP32 混合及共享专家。Attention、KV、模型其他计算、LM-head 和采样仍为 eager。
首次 prefill 即使只有一个 token 也不进入 decode 图；输出 4 token 对应 3 次 decode replay。

`none` 为默认，`moe` 需要同时指定 `--device-route-decode`。`full` 暂未实现，明确拒绝，
不能把 MoE 子图报告为 full-model graph。服务继续设置 `--enforce-eager` 和 compilation NONE。

## 1. 先运行不加载模型的原生探针

在目标 Linux NPU 容器内使用已经通过 fix2 的同一 `.so`。下面路径是已 PASS 的 fix2 库示例；
若现有 fix2 库在其他位置，所有命令统一替换为那个实际路径。本轮没有 C++ 改动，
无需重编译已经 PASS 的 fix2 库；不自动构建、重打包或安装。
先确认物理设备选择值 1 空闲；探针遇到 busy/unknown 会停止，不终止其他进程。

```bash
python -u tools/validate_vq2a8_v4_graph.py \
  --library build/vq2a8-ascendc-v4-device-route-fix2/libvq2a8_ascendc.so \
  --physical-npu 1 --timeout-s 300 \
  --report-dir reports/v4-graph-native
```

仅检查命令、不导入 torch/vLLM、不创建报告目录：

```bash
python tools/validate_vq2a8_v4_graph.py \
  --library build/vq2a8-ascendc-v4-device-route-fix2/libvq2a8_ascendc.so \
  --plan-only --queue-lifetime
```

探针在有界子进程中执行真实 `torch.npu.NPUGraph`；超时只清理该子进程会话。
依次验证：

- native `ResidentBank.select` 图；固定 K=512/1024、routes=1/6。
- `select → 原 rowwise preparation → native project` 图，逐张量字节一致。
- H=512 的完整合成 MoE，hash top-k 1/6、非 hash top-k 1、4 个 packed resident slots、
  8-ID 稀疏 lookup，以及原共享专家 gate/up/SwiGLU/down 实现。
- 同一静态输入地址上的 A→B→A、仅改 ID、仅改 hidden、重复/置换路由、
  非法 ID（含大 int64）、NaN、valid→invalid→valid，以及旧输出不被重放覆写。

低层探针在显式专用 owner stream 内构造 bank 并捕获，保留 native 同流检查。
完整 MoE 使用生产 `prepare_graph/forward_graph` 路径：专用图流、仅元数据的新 bank、
复用原 resident payload、调用流到图流及返回方向的事件依赖和 allocator 保活。
不会在失败后静默换回 eager 并报告 PASS。

`summary.json` 只有匹配的真实子进程回执通过后，才设置
`device_execution_verified=true`、`graph_functional_verified=true`。
`full_model_verified`、`full_model_graph_verified`、`serving_verified`、
`performance_verified` 和 `timing_valid` 仍为 false。

## 2. 可选的有界异步生命周期回归

```bash
python -u tools/validate_vq2a8_v4_graph.py \
  --library build/vq2a8-ascendc-v4-device-route-fix2/libvq2a8_ascendc.so \
  --physical-npu 1 --timeout-s 300 \
  --queue-lifetime --queue-iterations 2049 \
  --report-dir reports/v4-graph-lifetime
```

在短检查后额外运行：

1. 2049 次完整 hash/top-k6 MoE 的生产跨流 replay，混合调用流上的 eager matmul。
2. 2049 次原生 pipeline 图 replay + eager matmul，不混入 grouped 的描述符 H2D。
3. 2049 次 pipeline 图 replay + 原 single/grouped/grouped-pipeline eager 投影。

循环内无显式逐轮 synchronize、Tensor host read 或值打印；只在阶段边界同步校验。
临时输入在后续 eager matmul 前释放，输出只保留首/末有限样本。原 grouped ABI 的
描述符传输仍可能阻塞，所以第 2 段独立保留；不能把“未调用 synchronize”等同于零等待。

报告记录 capture/replay/entry、生产 stream bridge 数、HBM allocated/reserved、RSS。
图不允许重捕获或增长 entry。内存采用固定 64 MiB 的合成探针增长容差，超出即 FAIL，
不会动态放宽预算。这个短期上界不是任意长服务稳定性的证明，也不是 43 层图池预算。
replay 次数不等于实测 task-queue slot 次数，不作队列 slot-wrap 或纯 kernel 性能声称。

## 3. 同引擎小规模 A/B

原生图与生命周期检查通过后再加载模型。对照是同引擎、同权重、同库的
`device_route_decode_eager` 与 `moe_graph`，不是旧 host-route batched 或 V3。
保留原有显存参数；下列 maxlen16/KV256 MiB 示例不自动降低 reserve。

```bash
python -u tools/accept_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-device-route-fix2/libvq2a8_ascendc.so \
  --physical-npu 1 --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --cache-reserve-gib 3 \
  --device-route-decode --decode-graph moe \
  --cases 10:4,1:4 --warmups 2 --repeats 5 \
  --output-dir reports/v4-moe-graph-ab
```

要求 tokens/logits bit-exact、finite、每层真实 decode replay 覆盖、测量期间 capture 增量为 0、
expert payload 无加载/驱逐/H2D。capture/warmup 不计入请求时间，不重复执行模型/KV 来暖 MoE 图。
replay 不执行原 Python native counter；逻辑 replay 证据不能冒充 profiler 实测 launch 数。
不为此重新运行 V1 全模型对照、重打包矩阵或无关版本一致性检查。

五对样本用于筛查；趋势明确后再用至少二十对样本重复确认。性能改善门槛只是预先约定的
工程判断，不是速度承诺；没有最新 TTFT/TPOT 时，不承诺毫秒数或加速比。

## 4. 标准 HTTP 服务

```bash
python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-device-route-fix2/libvq2a8_ascendc.so \
  --physical-npu 1 --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 \
  --device-route-decode --decode-graph moe --host 127.0.0.1 --port 8000
```

在服务声明图初始化完成后测量，首次真实请求不能 lazy capture。
图池在所有层累积的实际 allocated/reserved 高水位要单独核对；不把一层结果乘 43 当作实测。

```bash
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --prompt 'Hello' --max-tokens 4 --warmups 2 --repeats 5
```

确保 tokenizer 后输入加输出不超过配置的 maxlen16。HTTP TPOT 根据
`usage.completion_tokens`，不按 SSE chunk 数；这是客户端 delivery timing，
不能和离线 engine token-ready timing 混算。先记录成功次数、实际 token usage 和原始样本，
再用相同客户端/同卡/同 prompt/同库的 eager 服务作串行对照。

## 5. 停止与回滚

capture/replay 异常、bit-exact 不符、validity 失效、旧输出被覆写、意外重捕获、
内存超界或超时，立即停止候选阶段。保留失败阶段和完整日志，不通过删检查、
关闭 task queue、逐层同步或放宽精度来“通过”。

停止自己拥有的候选服务后，用同样的服务命令将 `--decode-graph moe` 改为
`--decode-graph none`（或删掉该参数），保留 `--device-route-decode`、相同 fix2 库、
权重与预算，即恢复现有 eager 基线。不在已出错的 runtime 内自动降级后继续接受 token，
不停止其他进程或重置整卡。
