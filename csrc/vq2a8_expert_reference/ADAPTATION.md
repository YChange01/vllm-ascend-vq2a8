# 专家内核替换审查与接入方案

后续实施状态：本方案的 codebook 重排路线已形成独立 **VQ2A8 算子 v2** 候选实现与模型接入代码，
见 [v2 使用说明](../vq2a8_ascendc_v2/README.md)。下面保留实施前的审查依据，不作为“当前仍未写接入代码”的说明。
当前尚无服务器 CANN 编译、NPU 数值或性能通过结论。

日期：2026-09-09。现有后端基线：`f989dec9b0c1a54e2609976b1b82bf2f5376f758`。
专家输入：本目录聊天恢复稿，不是经原作者核验的原始源码包。
本次修改只在本地专家目录内；未推 GitHub，未更改已有权重、模型 dispatch、构建脚本或已验收的 `.so`。

## 结论和当前状态

**可以作为替换基础，但不能直接替换 `.so`。当前仅完成 CPU 布局桥、审计/输出检查和确定的 host 计时修正。**
专家内核的寄存器 LUT、zN、较大 N/K tile、多级 ping-pong 值得保留；真实模型数学契约不能省略。
主要阻断为逐 K 的码本选择、FP32 输出修正、Torch/分组指针 ABI，以及缺失的原始 MIX 构建配方。
不能通过改库名或重新 `pip install -e .` 消除这些差异。

## 真实权重检查

只读扫描本机 `D:/projects/2-bits/models/vp2a8/experts_vq_ascend_v2/tp1/rank0`：

| 项目 | 结果 |
| --- | ---: |
| safetensors 层文件 | 43 |
| gate/up 矩阵，N4096/K4096 | 10,243 |
| down 矩阵，N4096/K2048 | 10,243 |
| 物理 K256 块 | 245,832 |
| 同块混合多个码本 ID | **245,822** |
| 每 ID 的列数不是 256 的矩阵 | **0** |

此扫描读取 tile-ID 张量与 packed shape，不代表已逐字节验证所有矩阵。
另对 layer3/expert0 的完整 gate/up、down 做了转换往返及解码 FP8 权重字节核对：
25,165,824 个解码权重字节均相同；没有 NPU 数值/性能结果。
这两路真实首 LUT 并非四标量笛卡尔积，不能用四级 W2 重新拟合。

实测 JSON：[layer3-expert0-bridge.json](evidence/layer3-expert0-bridge.json)。
本次完整参考目录 **63 个 CPU 测试通过**（含 4 个 host 计时源结构回归；不是 host C++ 编译测试），
新增 Python 文件 Ruff 检查通过。真实两矩阵审计本机约 2.7 秒；不代表任何 NPU 执行耗时。

两边都以一个 uint4 选择两个相邻 N 输出的 FP8 字节；专家 AIV 的 16 项双字节 LUT 可以表示任意 VQ pair。
当前 int32 `[N/2,K/8]` 沿 K 打包，专家 uint8 `[N/32,K/16,16,8]` 沿 N 打包；字节布局可无损转换。
但当前 `codebook_tile_ids[k]` 保存了列置换后的真实码本 ID，专家原代码仅按 `k/256` 选 LUT。
只换 packing 不处理 tile IDs 会使几乎所有块查错码本。

## 推荐适配路线

### 1. 保留现有量化定义，不先改为新的 K32 MXFP8 量化

现有数值顺序必须是：

```text
原 physical-K 输入
  → 原 RHT128/sign
  → 原 weight_scale 变换、原 bias_correction
  → 原每行 FP32 scale 与 E4M3FN 量化
  → [若采用排序路线，仅在此处重排 FP8 A 字节]
  → FP8 dot 得 FP32 累加
  → FP32 × row_scale + row_bias
  → 最后一次 BF16 舍入
```

不得先重排输入再做 RHT128，否则改变变换分组；不得将一般 FP32 行 scale 强行转 E8M0；
不得先落 BF16 再补 scale/bias，那会多一次不可逆舍入。
若沿用 `mad_mx`，第一个受控候选可令 A/B MX scale 均为 1，保留原 FP8 数据和 FP32 epilogue。
这仍需修改 AIC 输出和增加/融合 epilogue；当前专家函数只有 BF16 输出，不接受 row scale/bias。

### 2. K 顺序提供两条候选

| 路线 | 要改什么 | 取舍 |
| --- | --- | --- |
| preserve（CPU 桥默认） | zN 保留物理 K，AIV 增加逐 K tile-ID 选择 | 不额外重排 A；专家单 LUT K256 复用逻辑须重做 |
| codebook（实验） | stable argsort(tile_ids)，W code 列和准备完成的 A 同时 gather | 保留原专家 K256 LUT 复用；改变累加顺序，需要精度门禁 |

当前每 ID 恰好 256 列，因此排序路线不需要原 `perm`，也不需要重新训练/量化权重。
同一双射作用于 A、W 的 K 维，数学点积不变；不同 K 分组及 MAD tile 仍可能改变 FP32 误差。
不得关闭已有 exact 门禁、把不一致当作无损优化通过。先量化误差，再决定是否批准非逐位候选。

为尽量利用专家版吞吐设计，可优先上板评估 codebook 候选，并把 gather 融入激活准备的输出写回；
保留 preserve/现有后端作为比较基准。CPU 桥只给数据准备和证明方法，尚未实现该设备融合。
“权重字节无损”不等于“新内核浮点结果逐位相同”，两条路线都需要实机精度验证。

### 3. 权重只在缓存装载时转换，不能每个 token 拼接全部专家

当前缓存中每个 expert 的 tensor 是独立地址；专家示例按 `base + group_id × stride` 假设 B/LUT 连续。
建议改为设备 job 描述符/指针表，持有各 expert zN、LUT、A、FP32 row scale/bias 和 C 地址。
缓存布局需独立版本/哈希；不在每次 decode 使用 `torch.cat` 复制数十 MiB 权重，不新建全层展开 FP8 权重池。
原 G≤128 并非支持 256 全层专家；先只绑定活跃专家，prefill 超限明确分批。
分组数不是并行度：专家代码仍在每核心逐 G 循环；现有 decode 已为每层两次投影 launch，不能重复计算合并收益。

### 4. 保留现有集成约束

- 使用 Torch 当前 NPU stream、device guard、Torch 分配和异步 owner 保活；不能在热路径调用 standalone 的初始化、reset、H2D、独立 stream。
- 使用独立后端命名空间或新进程；当前 loader 不允许同进程注册第二套同名 op。
- 构建 manifest/receipt 要覆盖专家 AIC/AIV、tiling、桥/epilogue、Torch binding、MIX 命令及库 SHA256；旧预检回执不得复用。
- 当前模型 N4096/K4096 或 K2048 满足示例形状，但旧通用预检的小尺寸不满足 N/K%1024；必须有明确能力检查，不能静默 fallback。
- 专家固定 32 AIC，AIC/AIV 的循环步长和 wide 2×16 映射也写死。不能只将 launch 数裁成 28；应核验拓扑或两侧一起适配。

## 已确认问题与本次处理

| 问题 | 本次处理 / 仍需工作 |
| --- | --- |
| host 反复记录同一事件，最后一次间隔除以总次数 | 已修复为驻留输入、完整循环外的一对事件；有限正计时检查；明确不含模型准备/H2D/D2H |
| 没有输出数值判定 | 已加独立 `check_output.py`，显式容差，区分 FP32 Golden 与 BF16 舍入 Golden；尚无真实设备 output 可验证 |
| 直接布局不兼容 | 已加 `vq2_bridge.py` 及 safetensors 只读审计；不会写回/批量转换用户 checkpoint |
| 原模板类型从聊天中推断恢复 | 必须与原始文件核对，尤其 AIV RegTensor 和 AIC create_cbuf_matrix cast |
| Fixpipe `16<<34` 与注释 `2<<34` 分歧 | 未猜测更改；需对应 Ascend950/CANN 定义/原作者核实，做转换及舍入边界测试 |
| 缺少 AIC/AIV 编译、MIX 链接脚本 | 阻断实际编译；请提供专家原始源码包及构建命令 |
| 片上内存与事件同步 | 未凭静态阅读宣称硬件正确；保留原同步，专项测跨档位、空组、奇偶 tile 与重复执行 |

事件计时 API 定义的是两个事件之间的时间，不是多次记录的总和；修复按其起止/同步流程设计。
依据：[华为 aclrtEventElapsedTime 文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/apiref/appdevgapi/aclcppdevg_03_0084.html)。
该文档不证明本机 CANN9.1/Ascend950 上的编译和计时已经通过。

N6144 分支 UB 达 259264/262144 字节，small M49..64 复用 L1A；当前模型全是 N4096，不能靠它覆盖 N6144。
L0C 模式切换 fence 已存在；`wide` helper 也有 M≥480→local_M≥240 的前置条件，不应误报为必然无同步或下溢。

## 本地审计使用方法

在仓库根目录，使用已有 CPU Torch/NumPy/safetensors/ml_dtypes 环境。不要为运行这些 CPU 工具升级服务器 NumPy。

```bash
python -m unittest discover -s csrc/vq2a8_expert_reference/tests -p 'test_*.py' -v

python csrc/vq2a8_expert_reference/scripts/audit_vq2_bridge.py \
  /home/g00872988/vq2a8/experts_vq_ascend_v2/tp1/rank0/experts_vq_layer_3.safetensors \
  --projection gate_up --projection down --expert-index 0
```

第二条是服务器路径示例：只切片所选 expert，不读全层大权重；JSON 写标准输出。
返回成功只证明本次 CPU 布局核对通过，输出仍明确 `drop_in_replacement_ready=false`。
当前不提供生成新 `.so` 的命令，因为原始 AIC/AIV 构建和 MIX 链接步骤缺失。

在取得真实 kernel 输出后，可独立检查（以下零容差是显式严格比较示例，不是建议自动放宽精度）：

```bash
python csrc/vq2a8_expert_reference/scripts/check_output.py \
  --metadata input/metadata.json --output output/output_c.bin \
  --golden output/golden_c.bin --rtol 0 --atol 0
```

该工具按 BF16-rounded FP32 Golden 判定，并另报未舍入 FP32 误差；输入错误和数值不符均非零退出。
它不会验证 kernel 二进制来源、执行指令或模型质量，不能取代上板回执。

## 后续上板验收顺序

1. 原始工具链构建单算子，固定硬件/源码/库哈希，核实 MIX、拓扑、FP8 指令和输出格式。
2. 任意 16×2 VQ pair（非四标量特例）、M1/15/16/17/48/49/64/65/96/97/255/256/257/479/480/481、
   K2048/4096、N4096及独立N6144；穿插空组、多组跨档、奇偶tile、重复执行。
3. 真实 gate/up、down 单投影 → 六专家 → 完整 MoE 层；保留原 preparation、scale/bias 和路由加权顺序。
4. 原后端与新后端分进程比较 token/logits/finite/repeat；先 exact，若不一致不得自动放宽阈值。
5. 同机同 prompt 和缓存状态测 10:4、32:32、96:32 的 TTFT/TPOT、prepare/gather/epilogue 与 launch 数。
   先验可用性，无硬性速度门槛；不能把 standalone TFLOP/s 当模型速度。

当前 `batched` 的短测 TTFT≈1.035 s、TPOT≈0.301 s 是已有后端数据；本目录尚无新 NPU 性能。
真正切换服务默认 backend 必须在新库完成上述验证后进行，本次没有把桥的 CPU PASS 当作替换完成。
