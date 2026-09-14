# 公司专家 VQ2A8 算子：独立参考整理稿

更新：模型适配后的实现统一命名为 **VQ2A8 算子 v2**，代码、构建和使用方式见
[../vq2a8_ascendc_v2/README.md](../vq2a8_ascendc_v2/README.md)。本目录继续作为原始材料与 CPU 审计归档，
不是当前模型构建目标。以下“未接入”描述是初次整理时的状态；当前已提供显式 v2 接入代码，但仍未完成服务器编译/NPU 验收。

来源：用户于 2026-09-09 在对话中提供的九文件代码文本。
本目录用于原始材料归档、审阅和适配溯源；**参考文件本身不作为模型内核编译，也不替换现有默认后端**。
此归档按用户要求随 v2 接入代码同步至指定 GitHub 仓库，供服务器复现源文件哈希与审计。
用户称其为公司内部代码；保留收到的版权声明，不将其自动重新许可为仓库的 Apache-2.0。
仓库存放不改变原代码权利归属，也不构成新的开源授权。

2026-09-09 适配进展见 [替换审查与接入方案](ADAPTATION.md)：已补 CPU 字节布局桥、真实权重审计工具、
独立输出校验，并修正 host 计时。**这不是 NPU 后端替换完成；现有模型 dispatch 和已验收 `.so` 未改。**

## 文件

| 收到的文件 | 本目录位置 | 作用 |
| --- | --- | --- |
| data_utils.h | [data_utils.h](data_utils.h) | 文件读写、打印与 ACL 辅助宏 |
| hacl_rt.h | [hacl_rt.h](hacl_rt.h) | 原示例携带的 runtime 声明，不是本机 SDK 的替代品 |
| mat_fp4_aic.cce | [mat_fp4_aic.cce](mat_fp4_aic.cce) | Cube、MX scale、L1/L0 搬运与同步 |
| mat_fp4_aiv.cce | [mat_fp4_aiv.cce](mat_fp4_aiv.cce) | 寄存器 LUT 解码、UB/L1 双缓冲 |
| mat_fp4_host.cc | [mat_fp4_host.cc](mat_fp4_host.cc) | 独立 `.bin` 测试程序、tiling、launch 与计时 |
| mat_fp4_tiling.h | [mat_fp4_tiling.h](mat_fp4_tiling.h) | Host/AIC/AIV 共用分块规则 |
| scripts/dump_code_to_txt.py | [scripts/dump_code_to_txt.py](scripts/dump_code_to_txt.py) | 本地源码导出；不要公开内部源码 dump |
| scripts/gen_data.py | [scripts/gen_data.py](scripts/gen_data.py) | 合成 MXFP8/W2 输入与量化后 FP32 Golden |
| scripts/w2_layout.py | [scripts/w2_layout.py](scripts/w2_layout.py) | zN 打包、码本分组与 pair LUT |

另附 [比较报告](COMPARISON.md)、[CPU 测试](tests/test_reference_layout.py) 和
[C++ tiling 测试](tests/tiling_test.cpp)。这些附加文件不是专家原始交付物。

## 恢复说明：不是原始源码的逐字备份

对话文本已经丢失了 `#include <...>`、`RegTensor<...>`、`std::vector<...>` 等尖括号内容，
部分下划线宏被替换成了 Markdown 星号。因此这里是**按上下文恢复、重新排版的整理稿**，
不能用于证明专家原文件的字节哈希，也不能宣称已原样编译通过。

本次恢复/整理范围：

- `**aicore**`、`**gm**`、`**ubuf**`、`**cbuf**`、`**ca**`、`**cb**`、`**cc**`、
  `**global**`、`**VEC_SCOPE**`、编译条件与 Python dunder 名恢复为双下划线形式。
- 标准头文件按实际使用补齐；恢复 `CHECK_ACL` 续行、`__FILE__`、`__LINE__`、`template <typename T>`。
- Host 的 vector 元素类型按文件字节契约恢复：binary 为 `char`，group list 为 `int64_t`，
  A/B/table/scale 为 `uint8_t`，C 为 `OutputStorage=uint16_t`。
  `initializer_list`/溢出检查恢复为 `uint64_t`，容器尺寸为 `size_t`，文件读取为 `std::streamsize`。
- `pointer*` 恢复为 `pointer_`；`static_caststd::streamsize` 恢复为模板 cast；
  `**cbuf*_` 恢复为 `__cbuf__`；usage 中补回由参数解析明确对应的 `<K> <N>`。
- **推断待核实**：AIV table/value 使用 `RegTensor<uint16_t>`，packed/index/shift 使用
  `RegTensor<uint32_t>`，索引与输出分别重解释成 16/8 位寄存器；相关处已标注。
  AIC `create_cbuf_matrix` 丢失的 cast 类型暂恢复为 `uint32_t`，必须对照原文件/对应 CANN 确认。
- 64 位配置字段的常量/地址/维度 cast 按 `uint64_t` 恢复；`run_lut` 计数按形参恢复为 `uint16_t`。
- 合并了同类变量声明、缩进和部分解释性注释；host 的长篇输出压缩为同类摘要，并加入计时风险提示。
  核心循环、分块常量、布局、同步事件、`test_sync` 条件和计算路径按收到的文本保留。

**仍待核实、未猜测修正**：Fixpipe 的 `16 << 34`（原注释为 `2 << 34`）、
raw 指令字段、同步协议、量化策略和 `[aicore]` 方言写法。
这些属于语义/工具链事项，不是可以由文本清洗保证正确的内容。

后续已修正 host event 计时：输入驻留，起止事件包围整个测量 launch 循环，完成后总时间只除一次循环数。
此处主动改变了原整理稿的测试逻辑，非专家原样交付；不含 H2D/D2H，也不是整模型 TTFT/TPOT。

## 能运行什么

当前没有收到构建脚本、编译器参数、AIC/AIV MIX 链接命令、`mat_fp4.o` 或精度验收脚本。
本机是 Windows CPU 环境，没有目标 CANN 编译/Ascend950 执行环境。
**不能直接用 `tools/build_vq2a8_ascendc.py` 构建本目录，也不会产出现有同 ABI 的 `.so`。**

CPU Python 检查（在仓库根目录，使用已有 `numpy`、`ml_dtypes` 的环境）：

```bash
python -m unittest discover -s csrc/vq2a8_expert_reference/tests -p 'test_*.py' -v
```

初次整理时实测：12 个测试通过，覆盖字节位序、zN 往返及独立地址核对、码本轴序、任意 16 项双值 LUT、
K 列置换反例、E8M0 编码、空专家以及小型合成输入/Golden 文件。
环境：Python 3.14.7 / NumPy 2.4.3 / ml_dtypes 0.6.0，全部是 CPU 测试；
**没有修改服务器依赖，也不建议照此升级服务器 NumPy**。

C++ tiling 检查已提供，但本机未找到可用 C++ 编译器，尚未执行。
在有 C++17 编译器的 Linux 上，可在独立临时目录运行（不要加 `-DNDEBUG`）：

```bash
test_dir=$(mktemp -d /tmp/vq2-expert-tiling.XXXXXX)
c++ -std=c++17 -O2 csrc/vq2a8_expert_reference/tests/tiling_test.cpp -o "$test_dir/tiling_test"
"$test_dir/tiling_test"
```

它遍历 M=1..8192 的实际 header 分块函数，不包含设备同步或指令验证。
`gen_data.py` 在当前工作目录写入 `input/`、`output/`，应在新临时目录使用，避免覆盖已有数据。

## 接入前需要补齐

请保留/提供专家的原始源码压缩包及构建脚本，以取代本整理稿中的推断项。
后续应独立做编译、随机任意 VQ 码本精度、跨档位同步、修正后的计时和真实专家适配；
本轮没有注册新的模型后端，也没有把 CPU 测试标为 NPU/整模型验收通过。

新增文件：

- `scripts/vq2_bridge.py`：严格 uint4 pair→zN、保留逐列 tile ID，或实验性稳定 K 排序；保留 FP8 字节及 FP32 行 scale/bias。
- `scripts/audit_vq2_bridge.py`：只读 safetensors 审计，不重写模型，不启动 NPU。
- `scripts/check_output.py`：独立 BF16 输出/FP32 Golden 比较，显式指定误差阈值；不代表模型质量验证。
- `tests/test_vq2_bridge.py`、`tests/test_bridge_audit.py`、`tests/test_output_validation.py`、`tests/test_host_contract.py`：新增 CPU 回归。
