# GPT-oss 接入迁移说明

这份文档讨论的是：如果要把当前项目从 `Qwen3-30B-A3B + EAGLE + 自定义 MoE kernel` 迁移到 `GPT-oss`，需要改哪些文件、为什么要改，以及补实验时应该做哪些验证。

这里的重点不是“能不能先把模型 load 起来”，而是“怎样把现有的 `SPMLP + sddmm/spmm` 路径真正接到 GPT-oss 上，并保证正确性和性能实验可信”。

## 1. 已确认的 GPT-oss-20B 关键信息

我在 2026-07-09 下载了官方 `openai/gpt-oss-20b` 的 `config.json`，确认到以下配置：

- 架构名：`GptOssForCausalLM`
- `hidden_size = 2880`
- `intermediate_size = 2880`
- `num_hidden_layers = 24`
- `num_local_experts = 32`
- `num_experts_per_tok = 4`
- `experts_per_token = 4`
- `num_attention_heads = 64`
- `num_key_value_heads = 8`
- `head_dim = 64`
- `sliding_window = 128`
- `max_position_embeddings = 131072`
- `transformers_version = 4.55.0.dev0`
- `quantization_config.quant_method = "mxfp4"`

这里最关键的两个点是：

1. `hidden_size = 2880`，不是 128 的倍数。
2. 官方配置要求的 `transformers` 版本比当前仓库 `requirements.txt` 里的 `4.53.2` 更高。

## 2. 为什么当前代码不能直接套到 GPT-oss

### 2.1 `moe_test/main.py` 目前是 Qwen3 + EAGLE 专用入口

当前 `moe_test/main.py`：

- 写死了 Qwen3 基座模型路径。
- 写死了本地 EAGLE 草稿模型路径。
- `mtp` / `bm` / `bmeagle` 分支默认都走 `EaModel.from_pretrained(...)`。

这意味着如果你想直接换成 GPT-oss：

- 不是只改模型路径就够。
- 还要看 `EaModel` 和 EAGLE 的 wrapper 是否支持 `GptOssForCausalLM`。

### 2.2 `EaModel` 目前没有 GPT-oss 分支

`moe_test/EAGLE/eagle/model/ea_model.py` 里，`from_pretrained(...)` 当前只显式分支支持：

- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Qwen3MoeForCausalLM`
- `PhiMoEForCausalLM`
- 其他情况默认走 `Mixtral`

也就是说，`GptOssForCausalLM` 现在没有专门分支。若直接喂 GPT-oss，大概率不会按正确 wrapper 走。

### 2.3 `SPMLP` 当前假设的是 Qwen3-MoE 风格参数结构

`moe_test/mlp.py` 里的 `SPMLP` 目前假设 `origin_mlp` 有这些字段：

- `origin_mlp.gate`
- `origin_mlp.experts`
- `origin_mlp.top_k`
- `origin_mlp.num_experts`
- 每个 expert 有 `gate_proj / up_proj / down_proj / act_fn`

但 GPT-oss 的官方实现不是这个结构。

在本机另一套 `transformers` 源码里，`transformers/models/gpt_oss/modeling_gpt_oss.py` 的实现是：

- `GptOssMLP` 由 `router + experts` 组成。
- `router` 是 `GptOssTopKRouter`，字段是：
  - `top_k = config.num_experts_per_tok`
  - `num_experts = config.num_local_experts`
  - `weight`
  - `bias`
- `experts` 是 `GptOssExperts`，不是 `ModuleList`，而是打包参数：
  - `gate_up_proj`，shape 约为 `[num_experts, hidden_size, 2 * intermediate_size]`
  - `gate_up_proj_bias`
  - `down_proj`
  - `down_proj_bias`

这和当前 `SPMLP` 的假设差别很大。

### 2.4 GPT-oss 的 gated MLP 还有 bias / clamp / 自定义激活细节

官方 `GptOssExperts` 的核心逻辑是：

1. 先做一次 `gate_up_proj`。
2. 然后把结果按奇偶位拆成 `gate` 和 `up`。
3. 对 `gate` / `up` 做裁剪：
   - `gate = clamp(max=limit)`
   - `up = clamp(min=-limit, max=limit)`
4. 用：
   - `glu = gate * sigmoid(gate * alpha)`
   - `gated_output = (up + 1) * glu`
5. 再乘 `down_proj` 并加 `down_proj_bias`。

其中当前官方实现里：

- `alpha = 1.702`
- `limit = 7.0`

而当前 `moe_test/mlp.py` 的主路径没有这些 GPT-oss 特有步骤。

### 2.5 当前 kernel API 没有 bias 参数

当前 `mlp_kernel` 暴露给 Python 的接口是：

- `mlp_kernel.ops.sddmm(x, up, gate, ...)`
- `mlp_kernel.ops.spmm(x, down, ...)`

接口里没有：

- `up_bias`
- `gate_bias`
- `down_bias`

这意味着：

- 如果要接 GPT-oss，至少要像 `moe_src/test/test_oss.py` 那样，在 Python 侧对 `ir` / `mask_v` 手动加 `up_bias` 和 `gate_bias`。
- `down_proj_bias` 也必须补进去；当前 `test_oss.py` 保存了 `db`，但没有真正加到最终输出里，所以它只能算接近 GPT-oss 的原型，不是完整正式实现。

### 2.6 2880 不是 128 的倍数，当前 kernel 直接不安全

这是这次迁移最硬的结构约束。

当前 kernel 代码里和尺寸相关的假设包括：

- `moe_src/mlp_kernel/csrc/cuda/sddmm.cu`
  - `block_per_expert = e / kTileN`
  - 这里 `kTileN = 128`
  - 如果 `e = 2880`，会被截成 `22` 块，尾部 `64` 维直接丢掉
- `moe_src/mlp_kernel/csrc/cuda/spmm.cu`
  - 也有固定 tile 假设
- `moe_src/mlp_kernel/csrc/cuda/sddmm.h`
  - `static_assert(kTileK == 32, ...)`
  - `static_assert(kTileS >= 128, ...)`
- `moe_src/mlp_kernel/csrc/cuda/spmm.h`
  - `static_assert(kTileS >= 128, ...)`

另外，batch 维度也有现成约束：

- `moe_src/mlp_kernel/csrc/cuda/sddmm.cu` 和 `spmm.cu` 目前只 dispatch `M in {32, 64, 128}`。
- `moe_test/mlp.py` 则只在 `bs in {32, 64, 128, 256}` 时进入 kernel 路径，否则回退原始实现。

所以 GPT-oss 的 `2880` 维接入时，要么 padding，要么改 kernel。

## 3. 推荐的第一版方案：先走 padding，而不是直接改 kernel

当前仓库其实已经有一个很接近 GPT-oss 的参考脚本：`moe_src/test/test_oss.py`。

这个脚本里已经做了两件关键事：

- `self.original_expert_dim = 2880`
- `self.expert_w = 2944`

也就是把 2880 padding 到最近的 128 倍数 2944。

脚本里还做了：

- 对 `up_bias` / `gate_bias` 做 host 侧加法
- 对 `gate` / `up` 做 GPT-oss 风格 `clamp`
- 用 `alpha = 1.702`、`limit = 7.0`
- 在激活后把 padding 尾巴清零：
  - `ir[..., 2880:] = 0`
  - `mask_v[..., 2880:] = 0`
  - `activated_up[..., 2880:] = 0`

这说明当前仓库已经验证过一条可行思路：

- 模型逻辑上仍然按 2880 hidden 维工作。
- kernel 内部把 expert width 扩成 2944。
- padding 尾巴必须在合适的位置清零，避免伪值参与后续计算。

### 为什么推荐先 padding

因为这条路改动最少，也最容易先把实验跑起来。

- `2880` 对 `32` 是整除的，所以很多 K 维分块仍然能用。
- 真正不兼容的是 `128` 分块那一层。
- padding 到 `2944 = 23 * 128` 之后，可以最大程度复用现有 kernel。

代价也比较小：

- 额外 width 只有 `2944 - 2880 = 64`
- 相对开销约 `64 / 2880 = 2.22%`

## 4. 如果接入 GPT-oss，具体哪些文件要改

下面按“先做可用版，再做优化版”的顺序列。

### 4.1 `requirements.txt`

建议修改：

- 把 `transformers==4.53.2` 升到一个确认包含 `gpt_oss` 的版本。
- 至少要和官方 config 里显示的 `4.55.0.dev0` 同代，或者你本地验证过支持 `GptOssForCausalLM` 的版本。

原因：

- 当前 requirements 版本太老，未必自带 `transformers.models.gpt_oss.*`。
- 即使 `trust_remote_code=True` 能勉强 load，后续和 EAGLE wrapper、KV cache、MoE 模块替换的兼容性也会更差。

### 4.2 `moe_test/main.py`

最少需要改：

- 把写死的 `Qwen3-30B-A3B` / `EAGLE_model_path` 改成参数化。
- 增加一种 GPT-oss 入口，不要直接复用现有 Qwen3-only 假设。

建议分成两步：

1. 先加一个“不走 EAGLE，只加载 GPT-oss base model”的分支。
2. 等 `SPMLP` 和 kernel 适配稳定后，再考虑 EAGLE / MTP 接入。

否则一上来就把 GPT-oss 塞进 `EaModel`，定位问题会非常困难：你分不清是模型加载、KV cache、EAGLE wrapper，还是 MLP kernel 出错。

### 4.3 `moe_test/mlp.py`

这是接 GPT-oss 时改动最大的一个文件。

至少要补这几件事：

- 支持 `origin_mlp.router`，而不是只认 `origin_mlp.gate`。
- 支持从 `origin_mlp.experts.gate_up_proj` 解包出 `gate_proj` 和 `up_proj`。
- 支持 `gate_up_proj_bias` 和 `down_proj_bias`。
- 支持 GPT-oss 的：
  - `alpha = 1.702`
  - `limit = 7.0`
  - `(up + 1) * (gate * sigmoid(gate * alpha))`
- 支持 padding 宽度 `2944` 与原始宽度 `2880` 的映射。
- 支持把 padding 尾巴清零。

一个更稳妥的实现方式是：

- 不直接把现有 `SPMLP` 硬改成一个巨大的 if/else。
- 新增一个 `GptOssSPMLP`，单独处理 GPT-oss 的参数布局和 bias。

这样可以避免把当前 Qwen3-MoE 路径也改坏。

### 4.4 `moe_src/test/test_oss.py`

这个文件已经是现成参考，但建议继续改成“正式校验脚本”，至少补：

- 把 `down_proj_bias` 也纳入比较。
- 明确对齐官方 `GptOssExperts.forward(...)` 的完整数学表达式。
- 输出更系统的误差统计。

如果后面正式接 GPT-oss，建议再新建一个更明确的测试脚本，例如：

- `moe_src/test/test_gpt_oss.py`

职责是：

- 从真实 `GptOssMLP` 权重构造 kernel 输入。
- 和 HuggingFace 原始 `GptOssMLP` 前向逐项对齐。

### 4.5 `moe_src/mlp_kernel/csrc/cuda/sddmm.cu`

如果走 padding 方案，这里未必需要立即大改，但至少要确认：

- `e` 传入的是 padding 后的 `2944`，而不是原始 `2880`。
- 所有 index / buffer 分配都基于 padding 维度。

如果想做“原生支持 2880，不 padding”的版本，这个文件必须改，因为当前：

- `block_per_expert = e / 128`

会直接截掉尾块。

### 4.6 `moe_src/mlp_kernel/csrc/cuda/spmm.cu`

和 `sddmm.cu` 一样：

- padding 版可以尽量复用。
- 原生 2880 版必须显式处理尾块和边界 mask。

### 4.7 `moe_src/mlp_kernel/csrc/cuda/sddmm.h`

如果要支持非 128 对齐 expert width，这里要改：

- tile 布局假设
- shared memory layout
- `kTileS >= 128` / 128-lane 相关推导

### 4.8 `moe_src/mlp_kernel/csrc/cuda/spmm.h`

同样要检查：

- shared memory 布局
- `kTileS` 相关假设
- 是否允许 tail tile

### 4.9 `moe_src/mlp_kernel/csrc/cuda/cuda_api.cu`

当前 API 没有 bias 参数。

如果你不想一直在 Python 侧手动加 bias，有两种方案：

- 简单版：继续在 `moe_test/mlp.py` 侧处理 bias。
- 彻底版：扩展 `cuda_api.cu` / `torch_api.cpp` / `ops.py`，给 kernel 增加 `up_bias` / `gate_bias` / `down_bias` 参数。

从工程复杂度看，建议先走简单版。

### 4.10 `moe_src/mlp_kernel/csrc/torch_api.cpp` 与 `moe_src/mlp_kernel/ops.py`

如果你决定把 bias 也下沉到 C++ / CUDA 接口，这两个文件必须同步改：

- 修改 `m.def(...)` 的算子签名。
- 修改 Python wrapper 的调用参数。
- 修改 fake registration 的元函数检查。

### 4.11 `moe_test/EAGLE/eagle/model/ea_model.py`

只有在你确定“GPT-oss 也要走现有 EAGLE / MTP 框架”时才需要改。

要改的最少项包括：

- 给 `GptOssForCausalLM` 增加显式分支。
- 新增对应的 KV-cache wrapper。

这通常意味着还要新增至少一个文件：

- `moe_test/EAGLE/eagle/model/modeling_gpt_oss_kv.py`

否则 `EaModel` 根本不知道如何用 GPT-oss 的 attention / cache 结构。

## 5. GPT-oss 接入时还有一个常被忽略的问题：量化格式

官方 config 里写了：

- `quantization_config.quant_method = "mxfp4"`

而当前自定义 kernel 的输入假设是：

- expert 权重已经是普通的 `float16` tensor
- 可以直接用 `data_ptr<half>()` 喂给 CUDA kernel

这意味着你要先明确一件事：

- 你是要跑“原始量化 GPT-oss”
- 还是先把 expert 权重 materialize 成 `fp16/bf16` 再喂 kernel

现阶段更现实的方案通常是：

- 先加载一个能拿到普通浮点权重的 GPT-oss 版本
- 先把 kernel 路径跑通
- 最后再决定要不要研究和 `mxfp4` 共存

否则会同时引入“量化解包”和“kernel 迁移”两类变量，实验很难收敛。

## 6. 建议的迁移顺序

我建议按下面顺序做，而不是直接改 `EaModel`：

1. 升级环境，使 `transformers` 能正确加载 `GptOssForCausalLM`。
2. 写一个最小脚本，单独加载 GPT-oss 的一层 `GptOssMLP`，确认能拿到：
   - `router.weight / router.bias`
   - `experts.gate_up_proj / gate_up_proj_bias`
   - `experts.down_proj / down_proj_bias`
3. 参考 `moe_src/test/test_oss.py`，做一版 padding 到 `2944` 的 layer-level 正确性测试。
4. 在 `moe_test/mlp.py` 新增 `GptOssSPMLP`，先只替换 base model 的 MLP，不碰 EAGLE。
5. 先让 `hf + GptOssSPMLP` 的端到端生成跑通。
6. 再考虑要不要把 GPT-oss 接进 `EaModel` / EAGLE / MTP。
7. 只有当 padding 版已经稳定且性能不够时，再考虑原生 kernel 改造。

## 7. 建议补做的实验

补实验建议分成四组。

### 7.1 正确性实验

必须做：

- 随机输入下，对比 HuggingFace 原始 `GptOssMLP` 与 `GptOssSPMLP` 输出。
- 对比 router top-k 结果是否一致。
- 对比 padding 版在去掉尾部 `64` 维后，是否与原始 2880 维结果对齐。
- 单层误差统计：
  - `max abs diff`
  - `mean abs diff`
  - 超阈值元素占比

建议测试的 token 数：

- `bs = 32`
- `bs = 64`
- `bs = 128`

因为当前 kernel dispatch 只支持这些 batch tile。

### 7.2 性能实验

建议直接复用你现在在 `moe_src/test/test_torch_cuda.py` 和 `test_torch_cuda_hybrid.py` 的实验框架，但把 GPT-oss 的真实超参数代进去：

- `hidden_size = 2880`
- `intermediate_size = 2880`，kernel 内部先 padding 成 `2944`
- `num_local_experts = 32`
- `top_k = 4`

建议至少测：

- `batch_size = 64 / 128`
- `maxnnz = 4 / 8`
- `t_d = 1..32` 以及如果你希望和旧图保持一致，也可以继续扫到 `64`
- 三条线：
  - 自定义 CUDA
  - hybrid
  - `torch.bmm`

另外建议补一个对照：

- padding 版 kernel
- 原始 `torch.bmm`

如果后面你真的实现了“原生 2880” kernel，再加：

- native kernel
- padded kernel
- `torch.bmm`

### 7.3 稳定性实验

必须看：

- 单卡能否稳定反复跑。
- 多卡可见时是否会复现 `illegal memory access`。
- 长 prompt / 短 prompt 下行为是否一致。
- decode 阶段和 prefill 阶段是否都稳定。

因为当前 Qwen3 MTP 已经出现了“单卡能跑、多卡不稳”的现象，GPT-oss 迁移后不应默认它会自然消失。

### 7.4 端到端实验

如果你的目标不是只补 kernel microbenchmark，而是要有模型层面的结论，还建议补：

- 原始 HF GPT-oss
- HF + `torch.bmm` MLP
- HF + `GptOssSPMLP`

比较指标：

- 首 token 延迟
- 平均 decode 延迟
- 峰值显存
- 生成结果是否一致或足够接近

如果最终还要做 GPT-oss 的 EAGLE / MTP：

- 还需要额外比较 Acceptance Length
- 以及 draft / verify 路径对总吞吐的影响

## 8. 最后给一个结论性建议

如果目标是“尽快把 GPT-oss 接进来并补实验”，最现实的路线是：

1. 先不要直接改 kernel 支持原生 2880。
2. 先用 `2944` padding 路线把单层正确性和 microbenchmark 跑通。
3. 先在不接 EAGLE 的 base model 路径里验证 `GptOssSPMLP`。
4. 等这一步完全稳定后，再决定是否值得：
   - 改 `EaModel` / `modeling_gpt_oss_kv.py`
   - 改 native kernel 去掉 padding

这样可以把风险拆成三层：

- 模型结构适配
- kernel 数值正确性
- speculative decoding / 多卡运行时稳定性

按这个顺序推进，定位问题会清晰得多。
