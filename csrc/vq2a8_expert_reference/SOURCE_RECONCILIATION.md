# 专家源码恢复核对记录

日期：2026-09-09。此记录区分三件事：用户保存的聊天原文、此前整理的九个参考文件、
本轮为真实 VQ2A8 模型新增的 `../vq2a8_ascendc_v2/` 后端。三者不能混称“原样编译的专家算子”。

## 原始材料

已重新读取本目录 `code.txt`，文件当前为 **109,060 bytes**，包含 `[1/9]` 至 `[9/9]` 九段源码。
SHA256：`dcdb520a791f22329a859b7636581ea43f469c7d3dce5174000f7c5c4b181fbe`。
本轮没有修改该文件。
目录内 `.gitattributes` 为 `code.txt` 禁用换行转换，避免 Windows/Linux 拉取改变该原始材料的字节哈希。

`code.txt` 保存的是聊天形态文本，而不是编译器可直接消费的源码压缩包：仍有空白 `#include`、
缺模板参数的 `static_cast(...)` / `std::vector` / `RegTensor`，以及 Markdown 形态的
`**aicore**`、`**name**`。因此它可以核对数字、地址公式、循环和原始意图，
但不能恢复已经丢失的类型信息，也不能证明聊天之前文件的字节身份。

## 逐文件核对

| 文件 | 与 `code.txt` 的差异及结论 |
| --- | --- |
| `data_utils.h` | 补回 7 个标准头文件、`CHECK_ACL` 的续行反斜杠、`__FILE__`/`__LINE__`、`template<typename T>`、`static_cast<size_t>(written)`。枚举值和文件读写流程未变；两处单语句 `if` 去掉花括号属于排版整理。 |
| `hacl_rt.h` | 恢复头文件保护宏的下划线并压缩注释/排版。移除注释、空白、恢复 Markdown 宏后，C/C++ token 比对仅剩 `RT_MEMCPY_RESERVED` 后的可选尾逗号差异。结构体字段顺序、枚举常量和函数声明未改。它仍是示例附带声明，**不是 CANN 9.1 ABI 认证**；新 Torch 绑定不使用该头文件。 |
| `mat_fp4_host.cc` | 头文件、容器元素类型、`numeric_limits<T>` 和 cast 类型按上下文补齐；若干同类型变量合并声明，`tiling` 等局部变量缩写，长篇输出改为摘要。输入形状约束、内存容量常量、tiling 取值和参数顺序保持原方案。**计时循环另有主动修正，见下节。** |
| `mat_fp4_tiling.h` | 补 `cstdint` 与 `__aicore__`。四处临时 `mode` 变量内联到相同条件表达式，删除可选尾逗号和单语句花括号。分块常量、近半切分公式及分支条件未改。 |
| `scripts/dump_code_to_txt.py` | 恢复 Python dunder；增加内部源码提醒；合并两个 `continue` 条件，直接将 `os.path.join` 结果加入列表，`open(..., "r")` 改为默认读模式。过滤范围、排序、文件名和写出格式未改。 |
| `scripts/gen_data.py` | 恢复 `__future__`/`__name__`，格式化和调整注释；去掉 docstring 后 AST 比对仅有一条形状错误文案的空格变化。量化公式、随机数据生成、dtype、布局、文件名和 Golden 计算未改。Golden 仍是**量化后的 A/B 的 FP32 乘法**，不是未量化模型精度。 |
| `scripts/w2_layout.py` | 恢复 Python dunder、调整 docstring/排版。去掉 docstring 后 AST 完全一致：轴变换、bit shift、打包顺序和 pair LUT 生成逻辑未改。 |
| `mat_fp4_aic.cce` / `mat_fp4_aiv.cce` | 对应原文 559–999 行、1000–1313 行。原 1035–1043 行仍丢失 `RegTensor` 类型；恢复为 LUT/value 的 16-bit、packed/index/shift 的 32-bit、索引 16-bit reinterpret、输出 8-bit reinterpret，与 `0x000F000F` pair 索引构造一致，但仍须目标编译验证。原 688 行 `create_cbuf_matrix` cast 仍缺；原 711 行 `16<<34` 与 712 行注释 `2<<34` 冲突未猜改。两份恢复稿不直接编入 v2 模型目标。 |

Host 恢复的主要类型依据：`binary_` 为 `vector<char>`；group list 为 `vector<int64_t>`；
A/A-scale/B/table 为 `vector<uint8_t>`；C 为 `vector<OutputStorage>`（`uint16_t`）；
尺寸乘加和溢出检查为 `uint64_t`；容器大小为 `size_t`；文件读取长度为 `streamsize`；
运行流转换为 `aclrtStream`；设备 ID 最终为 `int32_t`。
这些是对上下文的明确恢复选择，不等同于找回原作者已丢失的模板文本。

## 原始计时缺陷与已经保留的修正

`code.txt` 的 `RunTest` 在每次测试迭代里重复记录同一对 start/end event，循环结束只查询一次时间，
之后又除以 `test_cycles`。同一对事件不能自动累计所有迭代，因此这个计时表达式没有证明得到
整个循环总时长；用它计算平均值/TFLOP/s 会产生误导。原循环还在各轮前重复 H2D 并同步。

此前整理的 `mat_fp4_host.cc` 已主动改成：

1. 输入先驻留，完成 warmup。
2. 循环外记录 start，提交整个测量循环，循环外记录 end。
3. 同步后取得一个批次总时长，检查其有限且为正，然后只除一次迭代次数。

这项修正本轮核对后继续保留，不能说成原文一字未改。所得数值是**驻留输入 kernel batch 的
摊销时间**，不含 H2D/D2H、量化准备、路由、调度，也不是模型 TTFT/TPOT。
`pass_ms / active_group_count` 只是摊销值，不是逐专家实测 latency。

## 恢复稿仍保留的限制

- `data_utils.h::ReadFile` 不检查 `sgetn` 是否完整读取；`CHECK_ACL` 只打印不抛错；
  INT8/UINT8 打印可能按字符显示。模型后端不依赖这些辅助函数；独立测试应使用已有
  `ReadExactFile` 和 `CheckAcl` 的失败处理，不能拿日志打印代替验证。
- `hacl_rt.h` 含手写 runtime ABI 声明；`flag = 0` 是 C++ 默认参数，虽有 `extern "C"`，
  不代表该文件本身可以作为纯 C 头文件消费。新绑定使用安装的 SDK 与 CANN 生成的 launch stub。
- 原 standalone 限制 G≤128、K/N 为 1024 正整数倍，并按固定 32 AIC 分核。
  模型适配使用所选 active jobs，不将模型的 256 个路由专家直接等同于一次 G=256 调用。
- `MatFP4GetWideTile` 的无符号 `local_m / 240 - 1` 依赖 `local_m≥240`。
  原上层只在 M≥480 并分成两个 lane 后调用，满足这个前置条件；不能脱离调用条件单独使用。
- 原 AIC `l0c_gm` 的执行字段为 `16 << 34`，注释另提 `2 << 34`。未将其猜改为另一个数值。
  模型候选绕开该恢复不确定项，采用有类型的 FP32 输出 API，再完成原 FP32 scale/bias 修正。

## 本轮模型候选不是静默替换

新目标 `libvq2a8_ascendc_v2.so` 使用 `../vq2a8_ascendc_v2/kernel.cpp` 和 `torch_binding.cpp`，
与已有 `libvq2a8_ascendc.so` 分开构建、注册和验收；不编译原 standalone `mat_fp4_host.cc`，
也不要求用户准备原始 `mat_fp4.o`。

原因是实际 VQ2A8 的任意 16×2 码本、逐列 `codebook_tile_ids`、RHT/sign/量化准备和
FP32 scale/bias 契约，不能静默换成示例生成器的四标量 level 或 K32 E8M0 量化。
桥接保留码本与激活量化字节；若按码本稳定重排 K，激活只在原量化完成后执行相同重排。
重排和新的矩阵归约会改变 FP32 加法顺序，因此必须重新做数值和整模型回归，不能复用旧库的
`baseline_exact` 通过结论。

## 构建依据与验证边界

采用安装的 CANN `ascendc_library` 负责 AIC/AIV 编译、MIX 1:2 元数据、设备对象合并和 host stub，
而非猜测 raw `ccec` / `ld.lld` 参数。构建框架核对依据是官方
[CANN asc-devkit](https://gitcode.com/cann/asc-devkit) 源码，本地核对提交
`0290f560c82a867528b8ecbdb1a20366a6760a64`，尤其是
`cmake/asc/ascendc.cmake`、`legacy_modules/function.cmake`、`bisheng_intf.cmake` 和
`util/extract_host_stub.py`。服务器实际构建使用**服务器安装的同版本 SDK**，不复制这份审计 checkout 的构建配方。

`tools/build_vq2a8_ascendc_v2.py` 记录 compiler version/help、SDK 配方及 exact SoC 配置哈希、
当前原文和 native 输入哈希。构建成功仅设置编译产物状态，device/model/native-instruction
验证位仍为 false。CPU AST/token 核对与 CPU 测试不等同于 CANN 编译、MIX 同步正确性或 NPU 性能结果。
