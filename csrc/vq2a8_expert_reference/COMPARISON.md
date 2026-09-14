# 专家实现与当前 VQ2A8 AscendC 后端对照

对照基准：仓库 `ascend950-vq2a8-v026`，HEAD `a261ed78ad9c25753fe581293a37e11ffd29f3c1`。
专家侧来自用户粘贴代码的[恢复整理稿](README.md)，不是经验证的原始源码包。
以下区分源码事实、数学推导与待实机验证事项；没有专家版同机性能数据。

本篇主体是历史基线 `a261ed78` 的比较，不代表后续 opt3 全部状态；当前替换审查见 [ADAPTATION.md](ADAPTATION.md)。
opt3 已有独立 L1 pipeline 候选。本次 host 计时修复、CPU 布局桥和输出检查器已补齐，原始 CCE/MIX 构建问题仍未解决。

## 结论

专家版更侧重大分块 grouped GEMM、寄存器查表和多级流水；当前版更侧重真实 VQ 模型契约、
小 M、PyTorch/NPU 生命周期及整模型接入。专家版值得吸收，但**不能直接替换现有 `.so`**。
它不是“FP4 矩阵直接乘 FP8”：文件名虽为 `mat_fp4`，权重存储平均 2 bit/标量，
先查 LUT 恢复 E4M3 字节，再进入 FP8/MXFP8 Cube 运算。

当前代码已经不是最初的逐元素标量解码版本：已有向量 Gather、单 tile 前瞻、分组投影。
此前用户实测日志中，decode 每 token 是 486 次逻辑投影、86 次实际 kernel launch；
不能继续用“486 次 launch”来估算专家版的潜在收益。

## 主要区别

| 维度 | 当前 `vq2a8_ascendc` | 专家 `mat_fp4` |
| --- | --- | --- |
| 编程接口 | AscendC `TPipe/LocalTensor/Gather/Mmad/Fixpipe` | 原始 CCE 指令、手工配置位字段，加 MicroAPI 寄存器解码 |
| 权重码本 | 任意 16 项二维 FP8 向量，每项两个相邻输出值 | 内核同样索引 16 项双值 LUT；生成器只构造四标量值的笛卡尔积特例 |
| packed 权重 | int32 `[N/2,K/8]`，沿 K 存八个 uint4 pair 索引 | uint8 `[G,N/32,K/16,16,8]`，同一 K 上相邻 N 的两个 pair 索引共一 byte |
| LUT 选择 | 每个物理 K 列有 `codebook_tile_ids`，支持已吸收的列置换 | 固定连续 K256×N32 共用一个 LUT，没有逐列 tile ID 输入 |
| 激活 | RHT/sign、weight scale/bias 准备；每行一个 FP32 动态 scale | 输入已经准备好的 E4M3；每 K32 一字节 E8M0，样例在 CPU 生成 |
| Cube / 输出 | 普通 FP8 dot，FP32 累加；AIV 行 scale/bias 修正后 BF16 | `mad_mx`，A 的 MX scale 参与乘积、B scale 常驻 1；直接 Fixpipe 写 BF16 |
| A 数据路径 | GM→UB，重排后 UB→L1→L0A | GM→L1 的 ND2NZ；scale 单独 DN2NZ→L0A_MX |
| 两个 AIV 分工 | 各处理 M 行半区与 N 输出半区 | 相同 N tile 的相邻 K 半区 |
| 分块 | 固定 M32/N32/K128；实际输入 M=1..32 | M 小/中/大多档，N128/192/256，L1 K512/1024，MAD K128/256 |
| 流水 | 单 L1，下一 K tile 在 UB 提前解码 | packed/decoded/LUT、L1、L0A/B ping-pong，部分档位 L0C 双缓冲 |
| 分组 | 每次 1..6 个 job，共同 N/K，分别持有 tensor 指针；job×N tile 扁平任务 | 每次 G=1..128，非累计 Mi、支持空组，A/C 拼接，B/LUT 各组连续；每核遍历 G |
| 形状与硬件 | M≤32、N%32=0、K%512=0，动态查询核心数及 1C:2V | Host K/N%1024=0，固定 32 AIC；更大的 M 在设备内部拆 tile |
| 模型集成 | 已有 torch 注册、当前流、owner 保活、repack、真实模型调用与短验收 | 独立 `.bin` runner；未提供 PyTorch 绑定、路由/激活/专家输出合并适配 |

当前实现证据：
[layout.h](../vq2a8_ascendc/layout.h)、[kernel.cpp](../vq2a8_ascendc/kernel.cpp)、
[torch_binding.cpp](../vq2a8_ascendc/torch_binding.cpp)、
[激活准备](../../vllm_ascend/quantization/vq2a8_activation.py)、
[repack](../../vllm_ascend/quantization/vq2a8_repack.py)。
专家实现证据：[AIV](mat_fp4_aiv.cce)、[AIC](mat_fp4_aic.cce)、[host](mat_fp4_host.cc)。

## 码本不是障碍，布局与数值契约才是

### 16 项 LUT 可表达任意二维 VQ 向量

对任意 nibble `q∈[0,15]`，专家 AIV 读取的是 `LUT[q]` 的两个字节。
内核没有强制 `LUT[q] = [levels[q&3], levels[q>>2]]`；这个约束只出现在测试数据生成器。
所以不能因为 `build_pair_lut()` 接收四个 level，就认为专家内核只能处理标量 W2。

CPU 测试用 32 个互异字节组成任意 16 项双值 LUT，验证了 nibble 解码/输出顺序。
这证明字节布局的表达能力，不证明损坏文本恢复后的 MicroAPI 指令已经编译或执行正确。

### 固定 K256 LUT 与当前列置换不能直接混用

当前 repack 将 `perm` 吸收到物理 K 列索引中，保留 `codebook_tile_ids[k]`。
一个物理 K256 块可能包含多个原始码本 tile；专家实现只按 `k/256` 取一个 LUT。
**只转换 int32→uint8 或 transpose 不足以接入。**

可行方向是恢复码本友好的 K 顺序，并同时以同一顺序重排准备好的 A；或者给专家解码增加
逐列码本选择。前者还要处理每专家不同的 perm、额外 activation gather 与缓存成本。
若重新组织 K 的累加次序，即使权重值不变，也必须复验浮点误差，不能承诺 logits 逐位相同。

### MXFP8 是新的激活量化方案，不是已有 scale 的另一种存法

当前近似计算：`BF16(row_scale * FP8_dot(prepared_A, decoded_W) + row_bias)`。
专家样例：`BF16(sum_k(A_fp8[k] * 2**e[k//32] * decoded_W[k]))`。

前者每行 scale 是一般 FP32 值；后者每 K32 scale 必须是 2 的整数次幂，且在归约中生效。
不能把原 FP32 scale 按字节转成 E8M0，也不能省略 RHT/sign、weight_scale 或 bias 修正。
专家 `gen_data.py` 仅在 CPU 上生成测试激活，没有实现现有模型激活准备热点的设备优化。

较低风险的首个候选可以保留当前量化准备与 FP32 输出修正，仅移植解码/流水；若暂用
MX 运算，可研究 scale=1 的控制路径，但仍需补 FP32 epilogue 并实测一致性。
不要先写 BF16，再做原来的 FP32 scale/bias 修正——那会提前引入一次不同的舍入。
真正切换到 K32 MXFP8 应作为独立数值方案验收，不能称为无损性能优化。

## 专家版值得吸收的具体点

1. **寄存器 LUT + 向量解包。** 一次取 64 packed byte，构造 128 个 pair 索引，写出
   256 FP8 byte。LUT 驻留寄存器、每 K256 块复用，减少当前 UB gather/临时偏移访存。
   这是结构上的机会，不是未经实测的吞吐承诺。
2. **离线预排 zN。** 以 N0 连续、适配 Cube L0B 转置的格式存储权重，避免热路径反复
   做当前布局所需的解码后重排；同时必须保留真实 perm/码本语义。
3. **扩大 N/K tile、多级双缓冲。** L1 K 从当前 128 提升为 512/1024，可减少按 K 的
   循环/同步次数；N128/192/256 降低小 tile 管理成本，但会改变小 M 并行度及资源占用。
4. **A 直接 GM→L1、Cube 直接输出。** 可卸载当前 AIV 的 A 重排与输出搬运工作；
   前提是补齐真实 VQ epilogue，不能只为少一次搬运删掉数学操作。
5. **大 M 设备分块。** prefill 可在一次 grouped launch 内处理较大 Mi，减少当前按 ≤32 行
   切批的调用次数。M/N 联合分核更适合大 M；decode 是否更快必须单独测。

分组方面，专家版不是把 128 个专家自动并行化：代码仍逐组循环。
当前版已合并六个活跃专家，单 token 通常每层 gate/up 和 down 各一次 launch；
专家版主要进一步收益应从 tile/搬运/准备/大 prefill 中寻找，不能再把六专家合并计算一遍。
模型存在 256 个专家时，G≤128 的示例接口也不能直接把整个层当一个 full-group 传入；
应明确只传活跃子集、分组分批，或扩展接口并测试。

## 必须先处理的风险

### 1. 性能计时口径有明显缺陷

[host 的 RunTest](mat_fp4_host.cc) **原整理稿**在 `test_cycles` 循环内部反复记录同一对 start/end event，
最后只调用一次 `ElapsedMillisecondsTo()`，再除以 `test_cycles`。
该代码没有保存或累加每轮时间，也没有在整个循环外记录总区间。
如果 event 返回最后一次记录间隔，就会把单次延迟再除以循环次数，夸大同倍数 TFLOP/s；
如果目标运行时要求其他 event 复用方式，也不能假定当前数值是有效的均值。
`test_cycles=1` 不触发“重复除法”问题，但仍不是完整性能评测。

这里依据源码指出计时错误风险；公开的
[AscendCL event 计时文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/apiref/appdevgapi/aclcppdevg_03_0084.html)
定义的是两个 event 之间的间隔，不是多次记录的样本总和。该文档不是本机 9.1 版本的
event 复用保证，具体复用约束仍须核对匹配 SDK。

**现已修正**：输入驻留，起止事件分别放在整个 launch 循环外，结束同步后检查总时间有限且为正，再除一次次数。
报告明确是 resident kernel batch average，不含准备/H2D/D2H，不是单次样本 median/p95，也不是模型端到端耗时。
原 runner 每轮重新 H2D A、scale、B 的做法已从测量循环移除；`pass_ms/active_groups` 仍只是平均分摊。
CPU 源结构回归已加入，目标 CANN 编译和设备 event 计时仍待实机确认。

### 2. 编译文本不完整，不能把整理稿当官方实现验证

原始标准头、模板实参、cast 类型、CCE 宏损坏；恢复推断详见 README。
缺少编译/MIX 链接脚本；host 所需 `mat_fp4.o` 也未提供。
现有 `.so` 注册 `vq2a8_ascendc::projection/grouped_projection`，专家 runner 用 raw runtime
注册 `mat_fp4`，参数 ABI、分配方式和流管理均不同。

### 3. 输出转换字段存在代码/注释分歧

AIC `l0c_gm()` 生效值为 `16 << 34`，原注释候选为 `2 << 34` 并标注 F322BF16/RINT。
本轮未擅改，须用匹配 Ascend950/CANN ISA 确认，并测舍入边界、正负数和小量值。
仅看到输出指针为 BF16 不能证明转换模式正确。

### 4. 多级同步、接近满容量的片上内存需要专项验证

专家代码手工规划 256 KiB UB、两个 256 KiB L1 槽、L0A/B 各 64 KiB、L0C 256 KiB。
按 host 公式，N6144 的 UB 最大为 259264 byte，已接近 262144 byte 容量；
N6144 small 档 A+B 占满单个 L1 槽，M49..64 使用已消费 A 区域存 scale。
这类优化有价值，但更依赖精确的传输范围和事件保护。

需测 M=1/15/16/17/48/49/64/65/96/97/255/256/257/479/480/481，
以及 small→middle→wide→small、不同奇偶 tile 数、夹空专家、重复执行。
特别核对 scale 的完整 K pitch、L0C 单/双缓冲模式切换、跨组 ping-pong/event 收尾。
`test_sync` 会跳过部分 scale 初始化/加载，只能用于专门诊断，不能开启后验收数值。
源码中的 4 KiB L0B_MX 映射和 32 核常量须在目标子型号上核实；本轮未证明兼容全部 950。

### 5. 测试数据覆盖和完整性不足以作为模型验收

生成器用随机四 scalar level，尚未覆盖任意 VQ pair、真实码本、列置换、RHT、bias 和模型路由。
Golden 是量化后 A/B 的 FP32 GEMM，不是原模型质量基准。
Host 只写 `output_c.bin`，没有读取 Golden 并做误差断言；“run completed”不是精度 PASS。
现已另加 `scripts/check_output.py`，需要在 host 执行后单独调用；它不自动证明模型质量或 CCE 执行来源。
布局函数还会先将输入转换成 uint8 再检查范围，生产导入器应另做原始类型/值域检查，
不能将这些便利生成函数直接用作不受信任 checkpoint 的校验器。

## 接下来的合理路线（本轮未自动实施）

1. 获取原始源码与构建命令，核实所有恢复项，保持两个后端独立。
2. 修复计时并补单算子 Golden 校验，用任意 16 项 VQ 双值 LUT、空组和跨档位测试建立专家基线。
3. 增加字节精确保真的布局适配及 perm/activation 对应验证；沿用当前激活数值契约，先比较
   单专家 gate/up、down，再比较六专家与整层输出。
4. 接入 torch 当前流/设备 guard/指针生命周期/缓存，分别报告单核事件时、激活准备、H2D、
   prefill/decode 端到端时间。先实测再决定是否采用 MXFP8 新数值方案。

首选借鉴顺序：寄存器 LUT 与 zN 布局 → 大分块/多级流水 → 大 M grouped 路径 → MXFP8 数值方案。
保持当前实现作为可运行对照，不能把专家独立算子 TFLOP/s 当作整模型加速结果。
