# V4 device-route 图模式实施计划

后续新增的短上下文、按位置捕获的 `decoder` 图采用独立开关，并不启用通用
`FULL_DECODE_ONLY` 引擎图；范围与实机验收见 [三项优化验证指南](vq2a8_v4_perf3.md)。

状态：MoE 子图及接入障碍已按本文落地，默认关闭。
CPU 合约/数值回归与 NPU 硬件验收严格分开：本地没有 NPU，不能把已实现或
CPU 测试通过写成图捕获、服务或性能实测通过。`FULL_DECODE_ONLY` 仍是条件性后续阶段。
现有 CLI 和实机验证步骤见 [复现文档](vq2a8_v4_graph_reproduce.md)。

基于提交 `5f5a86634` 的只读审查，以及用户反馈：fix2 的 queue-lifetime
小测试已 PASS，HTTP 请求可完成且时延有所下降。尚未收到修复后的完整
TTFT/TPOT 数据，因此不预设图模式的目标毫秒数或加速比。

## 1. 结论与边界

按以下顺序实施，保留每阶段独立对照和停止条件：

1. 冻结当前 V4 full-resident + device-route + fix2 eager 基线。
2. 验证原生 NPUGraph 能真实捕获现有 native select/project。
3. 实现单 token、完整 MoE 子图，仍让 vLLM 引擎、Attention/KV 和采样走 eager。
4. 完成数值、生命周期、同引擎 A/B 和 HTTP 验收。
5. 根据实测瓶颈，再接 vLLM 的 `FULL_DECODE_ONLY`。

第一阶段固定物理设备选择值 1、TP1、并发 1、BF16 roots、输入 10 / 输出 4、
max-model-len 16、KV 256 MiB，保留当前显存参数。权重和专家 artifact 不重打包，
不切换 V3、FWHT、投影数学或权重格式，不开启异步调度。

第一阶段的比较对象是同库的 `device_route_decode_eager`，不是旧 CPU 路由 batched。
这才能单独测出图重放收益。若必须修改 native binding，A/B 两边均使用同一新库；
旧 fix2 库保留作恢复基线。

10-token prefill 暂时保持 eager；图初版主要改善 TPOT，不承诺 TTFT 同幅下降。
`FULL_DECODE_ONLY` 在本仓库包裹的是 model.forward，LM-head、采样、调度和 HTTP
仍在其外，不能称为整个请求端到端都在一张图内。

## 2. 实现前审查基线与代码约束

下表的行为及行号以实现前提交 `5f5a86634` 为准；落地后的接入与使用方式见复现文档。

| 代码位置 | 实现前行为 | 图模式处理原则 |
| --- | --- | --- |
| `tools/serve_vq2a8_v4.py` | enforce-eager、compilation NONE | MoE 子图阶段保持不变；整模型阶段才修改引擎配置 |
| `vllm_ascend/quantization/vq2a8_offline.py:107`、`:223` | 配置与校验限定 eager/NONE | 子图阶段保持引擎约束；FULL 只对显式 V4 组合开放白名单 |
| `vllm_ascend/platform.py:58` | 禁用 DeepSeek V4 breakable PIECEWISE | 不是禁用所有图；不能为此直接删除平台保护 |
| `csrc/vq2a8_ascendc/resident_binding.cpp:95`、`:203` | bank 保存构造 stream，并强制同流调用 | 捕获流必须与 owner 契约一致，禁止删检查或只替换 launch stream |
| `csrc/vq2a8_ascendc/resident_binding.cpp:142` | select/project 用 RunOpApi，输出由 at::empty 创建 | 实测图分配器与 native launch 的捕获；提交回调保活不等于整个图保活 |
| `csrc/vq2a8_ascendc/torch_binding.cpp:111` | grouped 在 CPU 建描述符并阻塞 H2D | 保留在 eager prefill，不将其直接纳入第一阶段图 |
| `vllm_ascend/quantization/vq2a8_activation.py:142` | Hadamard 首用创建，缓存仅一个 key | 捕获前完成常量准备；检查 gate_up/down block，必要时按 projection 持有常量 |
| `vllm_ascend/quantization/vq2a8_optimization.py:98` | validity 是 Python Tensor 引用重绑 | 图输出使用固定地址、每步更新的 device validity，不能重放旧的 Python 状态 |
| `vllm_ascend/quantization/vq2a8_v4_device_route.py:118` | native/route 统计在 Python 增加 | capture 与 replay 单独计数，图外维护逻辑覆盖，不能冒充硬件实测 launch |
| `vllm_ascend/patch/worker/vq2a8_offline_model.py:243`、`:436` | 非空 attn_metadata 触发首次切换和 synchronize | dummy capture 也可能非空；改用明确的初始化/预热/捕获/真实请求状态 |
| `vllm_ascend/patch/worker/vq2a8_offline_model.py:476` | 在采样前汇总 validity，执行一次 bool | 第一阶段保留在图外，不删校验，也不新增逐层 bool |

`vq2a8_execution.py` 的普通诊断 forward 有层前后同步。图只能包无诊断同步的纯计算，
不能把整个现有外层 forward 原封不动套进 capture。

固定 shape 不代表固定专家。图中必须每次从设备 input_ids/hidden 重新计算路由，
并让 native kernel 读取本次 slots，不能把第一次 capture 的专家选择固化。

## 3. 分阶段实施

### P0：冻结基线与确认收益空间

- 保存代码 commit、库 SHA、模型/artifact 路径、设备、启动参数和真实 serving 结果。
- 同一容器客户端绕过代理，temperature=0，先确认成功 5/5；每请求输入/输出 10/4，
  五次合计输入/输出 50/20（不是单请求长度）。
- 首次初始化和 warmup 不计入稳定性能。先沿用 2 次 warmup / 5 次实测探索，
  再用至少 20 对样本确认趋势。
- 采集一个短 decode 的 CPU/NPU 时间线，分辨 Python 提交空隙、kernel 执行、
  Attention、MoE、最终 validity/采样等待。NPU event 区间包含提交空隙，不能称为纯 kernel 时间。
- 已通过的同一 fix2 库不必因制定图计划而再次重跑全部验收；代码/库变化时复跑相关回归。

交付：当前 eager 基线回执、短 profile、相同条件的复测命令。

### P1：小规模原生捕获探针，不加载完整模型

新增 `tools/validate_vq2a8_v4_graph.py`，复用小型合成 fixture，设置独立子进程超时。

按顺序验证：

1. `ResidentBank.select` 捕获与重放。
2. `select -> 原 rowwise preparation -> project` 捕获与重放。
3. 完整 MoE：路由、lookup、gate_up、SwiGLU、down、有序 FP32 加权归约及共享专家。

这里 routed gate/up 已合为一次 packed projection，不重新拆分或修改数学。

落地实现使用显式专用 owner stream，不依赖默认流可以捕获。所有 Hadamard、roots 和 bank
初始化均在 capture 外；在专用流上创建仅元数据的新 bank/lookup，复用原专家 payload；
建立输入 ready、计算完成的双向 event 依赖与跨流 allocator 保活。原 eager bank 不替换，
native 同流检查不删除；桥接和结果 clone 成本均计入性能。失败后不自动切换流。

先验证现有 `at::empty` 是否由图私有池安全管理。capture 期间可使用受支持的图分配器，
不笼统禁止所有张量分配，也不为未经观察的问题全面重写算子。
只有发现现有接口无法捕获或无法保证地址/所有权时，才追加最小 `select_out/project_out`
绑定和正确的 mutation schema；保持原 kernel 计算以及 fix2 的 RunOpApi 释放路径。

通过条件：同一图、同一静态输入地址，改变输入内容后输出真实改变且与 eager bit-exact。
至少覆盖 A -> B -> A、同 hidden 换 ID、同 ID 换 hidden、重复/置换槽位、top-k 1/6、
稀疏映射、无效 ID、NaN、有效 -> 无效 -> 有效；capture API 返回成功本身不是 PASS。

### P2：图安全的 MoE 计算与生命周期

新增独立 `vllm_ascend/quantization/vq2a8_v4_graph.py`，不导入 V3 实现。

把 device-route 分成三个职责：

- 初始化：常量、bank、静态 hidden/input_ids、输出和 validity、图及 pool 的强引用。
- 纯计算：固定结构 Tensor 运算，输出 `(moe_result, valid_out)`，无日志、host 数值读取或 lazy 初始化。
- 执行包装：明确识别 prefill/decode、检查 signature、复制动态输入内容、replay、移交结果与计数。

图中每次生成本步完整 validity，写入稳定缓冲；不能捕获上次 state.valid 的 Python 引用链。
模型边界在采样前继续合并本步各层、hidden 和 logits 的检查；非法输入不得产生被接受的 token。

静态 signature 包括层、device、dtype、shape、top-k、projection 几何和 owner stream。
token ID、专家 ID、position 的数值不得进入 graph-cache key。
同层初版最多一个图 entry；不允许热路径隐式重捕获、同图并发或递归复用。

prefill 与 decode 必须按真实执行阶段区分，不能仅凭 hidden.shape[0]==1：
输入只有 1 token 的首次 prefill 仍走 eager，输出 4 token 对应 3 次 decode replay。

图对象持有 bank/roots/lookup/常量及图存储到最后一次 replay 完成。
输出初版保持调用者可安全持有的语义，必要的 D2D copy 成本纳入计时；不能返回会在下一次
replay 悄悄覆写的输出而不声明所有权。失败后禁止继续 replay，保留原错误；同步失败时
不得冒险释放仍可能在用的存储。

图销毁、bank/权重释放按依赖顺序处理：阻止新调用，等待已提交工作完成，释放图和其状态，
最后释放 bank/payload。禁止用永久保存每次输入/回调的列表绕过生命周期问题。

### P3：接入 V4 服务，保持引擎 eager

已新增 `--decode-graph {none,moe}`，默认 `none`，传入
`vq2a8_offline.v4_decode_graph`。`moe` 必须同时满足 V4、device-route、TP1/B1。
不增加环境变量，不恢复包版本或全局 pip 一致性检查。

在专用初始化阶段调用 `prepare_v4_graphs()`，明确在专家常驻、所需 warmup/profile 完成后，
且在服务宣告图就绪、正式测量之前准备图。评审最小 worker/patch 接入点，不能依赖
“非空 attention metadata 就是真实请求”的启发式判断，也不能在用户第一次 decode 内 lazy capture。

MoE 准备用 scratch hidden/token IDs，不执行整模型预热来污染 KV、position 或压缩状态。
先验证代表性 hash 层和普通路由层，再逐步覆盖全部 43 层；初始化进度需显示层号、
capture 时间、allocated/reserved 增量，便于在单层失败时定位。

初版独立图池，不立即跨层共享。各层 output 至少活到消费结束，validity 活到模型边界，
共享池可能覆盖这些地址。先测全模型图池高水位，再考虑有序共享；不能仅按静态 Tensor
大小估算图额外显存，也不能直接把单层消耗乘 43 当最终实测值。

不自动降低 KV、reserve 或专家常驻范围来塞下图。超预算、资源不足或 capture 失败时
终止该候选，保留旧 eager 基线。某个不支持的请求走预先定义的 eager 分支可以记录为 bypass，
但图发生异常后不得静默回退并继续输出 token。

### P4：功能、稳定性和性能分别验收

| 层级 | 最小验证 | 硬门槛 |
| --- | --- | --- |
| CPU | fake graph backend、配置、阶段识别、计数、所有权、失败清理 | 关闭图时原路径不变；plan-only 不初始化 NPU；fake 测试不冒充硬件证明 |
| 合成 NPU | 动态输入/路由、非法输入、输出保留、allocator churn、混合 eager 操作 | bit-exact，kernel 随输入重放；无超时/死锁/断言；无逐轮同步掩盖问题 |
| 真实层 | hash 与普通路由层、真实 projection 几何 | IDs/权重/prepared FP8 bytes/scale/bias/output 正确；常量不在运行期重建 |
| 同引擎 A/B | 一次加载，两种执行模式，AB/BA 交替 | tokens/logits bit-exact；测量期 capture 增量 0；无 payload reload/eviction |
| HTTP | 先 5 次，再至少 20 次并重复测量 | 实际成功、token usage 正确、真实图覆盖、报告 TTFT/TPOT 与离散度 |

合成长期测试至少连续 2049 次 replay，混合普通 eager 算子和原 grouped prefill，
包含临时输入早释放、有限数量旧输出保留和 teardown；只在测试边界同步并核对结果。
记录稳定期图 entry/capture 数、HBM allocated/reserved、进程 RSS；不允许持续增长，
也不把 replay 次数等同实际队列槽复用次数。

真实模型先测 `10:4` 和 `1:4`；在 maxlen16 范围内可用 `12:4` 验证边界。
不为了增加样本立即改长上下文配置。至少 2 次 warmup，先 5 对探索，再至少 20 对确认；
warmup/capture 独立记录。若需要更长 decode 的研究，另行明确调整上下文及预算，
不与当前 10/4 数据混在同一个对照里。

新图必须维持同库 eager 的 tokens/logits bit-exact；不在这一轮同时引入可能改变舍入的
FX fusion、FWHT、不同归约顺序或新 GEMM 几何，也不遇到数值差异就自动放宽容差。

建议预先约定推进门槛：至少三轮对照中 TPOT median 稳定改善约 10%，且 TTFT 无
明显回退、无稳定性/显存回退，再考虑推广；这是工程判据，不是速度承诺。
5 次样本不足以作可靠 P99 判断。若收益落在噪声内，标注收益未证实，先看时间线，
不自动增加更多图模式。

新增最小证据字段：

- `requested_graph_mode`、`effective_graph_mode`、`graph_scope`、`baseline_mode`。
- 每层 `captures`、`replays`、`bypasses_by_reason`、`signature`，capture/warmup 与真实请求分开。
- `capture_s`、`capture_peak_reserved_bytes`、`pool_count`、`static_buffer_bytes`。
- 每请求 graph 快照差值；10/4 每层 decode replay 应为 3，prefill replay 为 0。
- `graph_functional_verified` 与 `graph_performance_target_met` 分开；MoE 阶段
  `full_model_graph_verified=false`。
- 原 native Python 计数不再冒充图中的真实 launch；逻辑覆盖、profiler 观测和设备数值证据分别标注。

### P5：有条件推进整模型 FULL_DECODE_ONLY

前四阶段稳定后，若剩余 host launch/Attention 空隙值得优化，再新增 `full` 选项。
复用本仓库已有 ACLGraphWrapper、uniform-decode dispatcher 和 capture 生命周期，
只捕获 batch/token size 1，不启用默认的一串 capture sizes。

必须先完成以下核验，不能直接删 enforce-eager：

1. 提前初始化 device-route；dummy/warmup/capture 不能触发 configure_runtime/synchronize。
2. 对齐原生 bank owner stream 与 runner capture stream，保留所有权和依赖。
3. 明确关闭内层 MoE 图，捕获其纯计算，避免未经验证的嵌套图。
4. 盘点并验证 input_ids、positions、slot_mapping、block table、seq_lens、RoPE、
   SAS/QLI、compressor 状态的固定地址及每步内容更新；覆盖 position 连续变化、
   请求切换、slot 复用和状态复位。
5. 区分图外 metadata 构建的 CPU 读取与图内同步，不见 `.item()` 就盲目重写。
   DSA 已有部分固定缓冲和 `copy_` 更新；`update_graph_params` 的空实现也不能独自证明失败，
   必须核查完整数据流并实测。
6. 捕获整模型会执行 Attention/KV 写入，使用 runner 正确的 dummy/scratch 状态并恢复，
   验证捕获前后真实请求 KV/压缩状态不受污染。
7. wrapper 当前调试地址检查主要看位置参数，而 model runner 用 kwargs；增加必要的
   kwargs/metadata 地址证据，不能把空地址检查视为稳定性证明。
8. 不同时引入 Npugraph_ex 数学优化。优先验证支持的纯运行时 ACLGraph 路径；
   若框架必须经 FX，则单独补自定义类/算子的 functional dispatcher、fake/meta
   与 mutation 约定，单独数值验收，不临时绕过错误。

本地 runner 的 `_use_aclgraph()` 还有 compilation mode 依赖，不能预先承诺
`mode=NONE + FULL_DECODE_ONLY` 可以工作；需打印最终 resolved compile/runtime mode
及真实 capture entries。现有 wrapper 的 replay 同步先保留，是否可消除是后续独立性能改动。
短上下文验收覆盖压缩 c4 的 3 -> 4、7 -> 8、11 -> 12 边界和请求槽复用；
dummy positions=127 与实际 maxlen16 的组合单独核验。c128 的实际边界不在本轮覆盖范围内。

LM-head、模型最终 validity 决策和 sampler 初版留图外；整模型 forward 图需要重新跑
对应数值与状态回归，不能用 MoE-only PASS 替代。

prefill 图不在本轮首个交付范围内。若后续 TTFT 仍是主要问题，再单独研究
10-token prefill 的设备路由和 grouped 描述符路径，不能假设 decode 图会自动覆盖它。

## 4. 实施分组与后续范围

| 提交 | 文件/范围 | 交付 |
| --- | --- | --- |
| 1 | 新 `tools/validate_vq2a8_v4_graph.py`、图 CPU/合成测试 | 动态输入 native capture 最小证明，无整模型加载 |
| 2 | 新 `vq2a8_v4_graph.py`；调整 `vq2a8_v4_device_route.py`、`vq2a8_activation.py`、`vq2a8_optimization.py` | 纯 MoE compute、固定 validity、常量、单流图生命周期 |
| 3 | `vq2a8_offline.py`、`vq2a8_execution_v4.py`、`patch/worker/vq2a8_offline_model.py`；必要的最小 worker 初始化接入 | 配置白名单、初始化/释放、语义 decode、模型边界校验；默认关闭 |
| 4 | V4 serve/benchmark/accept、相关测试、复现文档 | 显式 moe 开关、同引擎 eager/graph A/B、图证据和 HTTP 测速 |
| 5（条件性） | `worker/model_runner_v1.py`、`compilation/acl_graph.py`、相关 Attention metadata；按真实缺口修改 | FULL_DECODE_ONLY、size1、状态复位与无嵌套图 |

native `resident_binding.cpp` 或 `torch_binding.cpp` 只在 P1 证明有必要时变更；
不预设必须新写数学 kernel。需要新库时使用独立构建目录，eager/graph 对照同库，旧 fix2 不覆盖。

## 5. 停止与恢复

任一图未捕获 native 工作、动态路由不刷新、validity 失效、数值差异、输出被覆写、
异流契约破坏、意外重捕获、显存持续增长、超时/死锁/allocator 断言，立即停止该阶段。
报告失败位置和原始日志，不通过关闭 task queue、逐层同步、移除边界检查或放宽精度来“通过”。

恢复方式是停止自己拥有的候选服务，以原 fix2、`--device-route-decode`、相同权重与预算
重新启动 eager；不停止其他进程、不重置整卡、不把出错中的运行时自动切回 eager 继续服务。

## 6. 参考与证据范围

- [Ascend 图模式指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)：
  ACLGraph 运行时捕获与 Npugraph_ex 编译阶段不同；上游模型支持不能替代本 VQ2A8 自定义路径验收。
- [torch-npu v2.10.0 NPUGraph 实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/npu/graphs.py)：
  显式捕获流、图池与生命周期接口；实际 dev wheel/CANN/Ascend950 行为需目标机器验证。
- 本地审查由主代理与三个子代理分别覆盖 native、runner/Attention、验收与性能。
  本轮已修改运行代码并增加回归测试；尚未执行 NPU 新图捕获，用户之前已 PASS 的是
  eager 异步生命周期回归。图功能、图性能和整模型图的证据不得混用。
