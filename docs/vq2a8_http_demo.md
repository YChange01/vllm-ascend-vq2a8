# VQ2A8 本机 HTTP 体验服务

这个入口把已验收的离线引擎包在一个本机 HTTP 接口后面，不是通用 `vllm serve`，
不宣称生产服务、模型质量、长上下文或并发已验收。原离线路径及默认开关不变。
仅监听 `127.0.0.1`；默认 `batched`，TP1、单线程串行、BF16 root、总上下文不超过 128 token。
只支持非流式文本补全，暂不支持 chat template、聊天接口、工具调用或并发调度。

HTTP 层使用标准库，无新增 pip 依赖。[Python 文档](https://docs.python.org/3.11/library/http.server.html)
明确不建议将 `http.server` 用作生产服务器；不要将此入口代理到公网或用于生产。

## 终端一：启动

以下验收目录来自本次已通过的短测；后续应显式使用对应的 PASS 目录，不自动挑选最近一次目录。
使用此前通过环境检查的 Python。无需重新 repack、重新 pip 安装或重新编译算子。

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only

python -u tools/serve_vq2a8_demo.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-opt3/libvq2a8_ascendc.so \
  --acceptance-report reports/vq2a8-opt3-20260908T235516715328Z \
  --physical-npu 0 \
  --preset batched \
  --port 8000
```

启动时核对已验收 Python 源码与库 SHA256、原算子预检回执及模型路径；
复用原 QLI/SAS 元数据检查、严格权重加载，再进行一次 10 输入/4 输出的真实生成冒烟检查。
只在初始化和冒烟检查完成后打印 `HTTP_DEMO_READY`。根据之前的加载时间，预留约 2～5 分钟，
这是估计而非本入口的 NPU 实测。启动期间保持终端开启，不要同时在 NPU 0 跑验收。

出现库/源码不匹配时不要跳过检查；先核对所选报告，必要时重新执行优化验收。
端口已占用会在模型加载前报错，可改用 `--port 8001`；不会终止原有进程。

## 终端二：在同一台服务器访问

```bash
curl -sS http://127.0.0.1:8000/health | python -m json.tool

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vq2a8","prompt":"The answer to 1 + 1 is","max_tokens":4,"temperature":0,"ignore_eos":true}' \
  | python -m json.tool --no-ensure-ascii
```

看响应的 `choices[0].text`。这不是流式接口，会在生成结束并完成有限值检查后一次返回。
`vq2a8.generation_s` 包含每请求配置、生成、有限值检查和解码，不是 HTTP TTFT 或 TPOT。
第一个新提示可能需要加载尚未驻留的专家，不能把冷请求耗时当成热缓存性能。

想看稍长的续写，可以请求：

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vq2a8","prompt":"The capital of China is","max_tokens":16,"temperature":0}' \
  | python -m json.tool --no-ensure-ascii
```

这是原始文本续写，不会擅自拼接聊天模板。默认遇到 EOS 停止；
`ignore_eos=true` 仅用于固定输出长度的复现请求。
输入长度按 tokenizer 实际结果加 BOS 计数，输入加 `max_tokens` 超过 128 时返回 HTTP 400，不截断。
仅接受单个字符串、`n=1`、`temperature=0`、`stream=false`；不支持的参数明确报错。
短测通过只证明报告中列出的 case，其他文本/长度属于体验性探索。

## 停止与故障

- 终端一按 `Ctrl+C` 退出；不要清空系统缓存、重置 NPU 或杀掉其他进程。
- 引擎异常或非有限结果返回 HTTP 500，之后 `/health` 和生成返回 503，需检查终端 traceback 后重启。
- 输入错误返回 400，不使引擎失效。客户端断开不会中途切换 NPU 所在线程，当前有界请求会完成。
- HTTP 请求串行处理；生成时健康检查也会等待。它不是并发服务，不适合压力测试。
- 启动前只检查过去的离线证据；HTTP 层和任意提示的 NPU 实测仍需要本次在服务器完成。

## 本地验证范围

CPU 测试覆盖输入限制、BOS/上下文计数、验收来源绑定、线程归属、有限值检查、
EOS 开关、故障隔离以及真实 loopback HTTP 路由。推理使用测试替身，不计为 NPU/模型服务验收。

本轮 Windows CPU 验证：新增 HTTP 测试 48 项通过；包含相关优化、0.26 兼容与原生接口契约的
回归选择共 417 项通过、4 项跳过。Ruff 检查和格式检查通过。没有在本机加载 Ascend 模型。
