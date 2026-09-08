# VQ2A8 opt3：按顺序验收的优化候选

本轮是 **候选实现**，不是已实测的 Ascend950 性能结论。默认模型、原验收脚本的
baseline/compact 行为、BF16 root 和原 `grouped_projection` 入口继续保留。
不需要重新 repack，不改已经验收的权重，不卸载依赖。

## 本轮修改及边界

| 顺序 | 候选/入口 | 已实现 | 尚未证明或未实现 |
| --- | --- | --- | --- |
| 1 | `fast` | 每层一次 route ID 主机物化；合并索引上传；数值有效性保留在设备，计时外统一检查；批量 SwiGLU/slot 写入 | lazy cache 仍需 CPU 路由；hash 输入/专家越界检查仍保留同步；descriptor H2D 仍同步 |
| 2、3 | `batched` | 同一 prefill 的 token/expert assignment 整体分组，每专家最多 32 行，每次最多 6 个 projection；重复路由只计算一次、完整保留 top-k slots | 没有完整 device-only scheduler，也不是原生融合 down+weighted-reduce 算子 |
| 2 | `fwht` | 按 assignment 批量的两阶段 Triton-Ascend 准备：RHT128 butterfly、scale/bias 部分归约，再做每行 E4M3 量化；metadata 按专家而非按行复制 | CANN/Triton 编译与 NPU 数值待实测；FWHT 的累加顺序不同，不能预设 bit-exact |
| 4 | `pipeline` | 增加独立 `grouped_projection_pipeline` 入口；L1 A/B ping-pong，AIV 与 AIC 按槽握手，跨 group 排空 event | 保留 K128/M32/N32 与现有 UB Gather；本轮没有引入寄存器 LUT、大 tile 或 L0 双缓冲；尚未实测流水收益 |
| 5 | `prepare_graph` | 合并 batched+FWHT+pipeline；只捕获 decode 激活准备子图，每层最多两个固定地址条目，每次更新 activation/sign/scale/bias；缓存 miss 可观察 | **不是整模型 decode 图**；不捕获 lazy cache、CPU 路由或权重指针；图池显存/收益待设备实测 |
| 6 | `--profile` | 计时外 CPU/NPU trace；route、prepare、projection、SwiGLU、mix/shared、decoder、attention/root linear 范围 | 未凭猜测改 attention/KV/root 数学实现；根据实际 trace 再决定下一项融合 |

`pipeline` 基于 batched + 原 rowwise preparation，刻意独立于 FWHT 的数值变化，
以便 FWHT 未通过时仍可验收流水。`prepare_graph` 是组合候选。
shared expert 保留原 token chunk 的 GEMM 几何，避免混入额外的 BF16 舍入变化。
因此这些不是对外 serving 开关，而是单进程、单流、TP1、上下文不超过 128 的实验入口。

## 与 GPU、专家示例的关系

- 借鉴 GPU 的 assignment 合批、FWHT 准备和稳定地址重放思路，不套用 H200 TP4/并发吞吐数字作为 TP1 TPOT 目标。
- 借鉴专家示例的片上生产者/消费者双缓冲思路；没有直接复制其内部 CCE 源码。
- 不将真实 VQ 的 16 个二维码字压缩成 4 个标量 levels，不丢弃 `codebook_tile_ids`。
- 不将现有每行 FP32 scale/bias 改成专家示例的 K32 E8M0 MXFP8 规则；原 repack v2 数据继续适用。
- 内部参考目录 `csrc/vq2a8_expert_reference/` 不参与本轮构建，也不因本轮修改自动授权公开发布。

## 新服务器统一入口

先将本轮代码同步到服务器的同一个仓库，再执行：

```bash
cd /home/g00872988/vllm-ascend-vq2a8
python -u tools/accept_vq2a8_optimizations.py \
  --model /home/g00872988/vq2a8 \
  --physical-npu 0 \
  --soc Ascend950DT_9574 \
  --profile
```

SOC 必须与当前设备一致，上面的值来自已有服务器日志。可先加 `--plan-only` 查看步骤。
脚本会顺序执行环境检查、单独构建、原算子预检、新候选微预检、整模型 exact 对比、
AB/BA warmup 和计时、独立 profile。默认每 case 每候选分别有 2 次 warmup、5 次正式测量。
完整矩阵包含 5 个候选 × 3 个 case，可能运行数小时；每个子步骤默认 4 小时超时，
可以显式调整 `--timeout`。超时只终止本次创建的子进程组，不操作其他任务。

新库默认输出至 `build/vq2a8-ascendc-opt3/libvq2a8_ascendc.so`，
**不覆盖** `build/vq2a8-ascendc-v026/libvq2a8_ascendc.so`。
已经构建好时可用 `--library <新库路径>` 代替 `--soc`，跳过编译。

想先验证最小可运行路径，可只选 `--presets fast,batched,pipeline --cases 10:4`；
这不等于完整矩阵通过。不要在其他 NPU 作业同时运行时将波动归因于某个代码优化。

## 报告与判定

报告位于 `reports/vq2a8-opt3-<UTC时间>/`：

- `run.json`：各子步骤命令、退出码、超时、错误；完整输出分别在同名 `.log`。
- `result/optimization-preflight.json`：pipeline 的跨 job/permutation/repeat 精确对比，
  FWHT 数值观察，以及准备子图更换输入/专家 metadata 后与 eager 的精确对比。
- `result/summary.json` / `summary.txt`：每 case/preset 的 baseline 和 candidate TTFT、TPOT、
  E2E 分布及样本，cache load/eviction、projection 次数、row histogram、图 capture/replay/bypass 计数。
- `result/profile-*/`：各通过候选、各 case 的独立 CPU/NPU trace 和 profile 结果。
- `.safetensors`：计时外的 baseline/candidate/repeat logits，便于逐项回归追溯。

数值门槛仍是 token 相同、logits **逐字节**相同、重复结果相同、43 层都实际执行，
不是独立高精度参考或模型质量验证。FWHT 若改变任意 logits 位，标记 `REJECTED_NUMERICAL`，
停止该候选的计时但保留证据；不会悄悄放宽误差或者回退。设备/编译异常则停止本进程并保留完整 traceback。

不设硬性速度门槛。`PASS` 仅表示相应候选数值门槛与测量矩阵完成，
**不表示速度提升、原生指令取证通过、整模型图通过或 serving 可用**。
正式样本若产生新 graph capture 会报错；应增加 warmup 后重测。
热 cache 和冷 cache 由样本的 load/eviction 区分，不把磁盘加载假称为 kernel 时间。
host timing 和 NPU event span 都含各自区间内的提交空隙，真正 kernel 归因以 trace 为准。

## 仍需硬件后续工作的项目

1. 先跑当前新增路径的 NPU 编译、微预检和 exact 模型回归；本地 Windows CPU 测试不能替代它。
2. 根据 trace 确定 descriptor 同步、寄存器查表、大 tile、down/reduce 融合的实际占比，分别做下一轮内核实验。
3. 整模型 ACL graph 需先解决动态专家 cache、可变路由、跨 replay buffer/指针生命周期；本轮不直接开启。
4. attention/root 优化保持单独数值验收。只有前述数据齐全后，才有依据扩大优化范围。

## 本地验证记录

Windows / Python 3.14.7 / PyTorch 2.10.0 CPU，仅运行 CPU 契约测试：

- 本轮相关测试选择：440 passed。
- VQ2A8 全量 CPU 测试选择：1082 passed、211 skipped、2 failed。
- 两项失败分别是 `test_inverse_rope_fp8_midpoint_regression_and_bounded_diagnostics`
  和 `test_rounding_probe_requires_exact_inputs_scales_and_fp32_before_weight_gates`；
  相关 root FP8 源码及测试与 HEAD 相同，本轮没有修改，也没有降低其门槛。
- Ruff 检查通过；Python/C++ 已格式化；`git diff --check` 通过。
- 没有 Ascend950/CANN 本地编译、NPU 模型实测、速度提升或 native instruction 取证结论。
