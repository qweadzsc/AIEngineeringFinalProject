# moe_test 说明

`moe_test` 不是训练框架，而是一个面向生成实验的轻量 benchmark harness。它目前绑定的是：

- 基座模型：`Qwen3-30B-A3B`
- 草稿模型：本地 `EAGLE-3`
- 可选的 MLP 后端：原始 HuggingFace / EAGLE、`SPMLP + 自定义 CUDA kernel`、`SPMLP + torch.bmm`

目录里的主入口是 `main.py`，其职责是加载指定方法、跑若干条 prompt，并统计延迟、Acceptance Length 和峰值显存。

## 1. 目录里主要文件在做什么

- `main.py`：当前推荐入口。
- `mlp.py`：定义 `SPMLP`，把稀疏 MoE MLP 改写成自定义 CUDA 路径或 `torch.bmm` 路径。
- `data.py`：解析 benchmark 数据集并组 prompt。
- `run.sh`：默认快捷命令，当前是单卡 `mtp`。
- `main_old.py`：更早的原型脚本，仅供参考，不建议继续使用。
- `EAGLE/`：vendor 进来的 EAGLE 代码。

## 2. `main.py` 在跑什么

`main.py` 接收三个参数：

- `--dataset`：数据集编号，对应 `['alpaca', 'commonsense_qa', 'gsm8k', 'hellaswag', 'piqa', 'siqa', 'sst2', 'sum']`
- `--method`：`hf` / `eagle` / `mtp` / `deepspeed` / `bm` / `bmeagle`
- `--num_prompts`：要处理多少条 prompt

主流程如下：

1. 用 `resolve_dataset_path(...)` 找到 benchmark 数据。
2. 构造 `CustomTextDataset`。
3. 写死基座模型路径 `/share/public/public_models/Qwen3-30B-A3B`。
4. 写死草稿模型路径 `/share/zhouyongkang/models/qwen3_30b_moe_eagle3`。
5. 根据 `--method` 分支：
   - `hf`：直接加载 HuggingFace 模型并 `generate(...)`
   - `eagle`：加载 `EaModel` 并 `eagenerate(...)`
   - `mtp`：加载 `EaModel`，把每层 `layer.mlp` 换成 `SPMLP(..., forward_mode='main')`，然后 `eagenerate(...)`
   - `bm`：加载 `EaModel`，把每层 `layer.mlp` 换成 `SPMLP(..., forward_mode='bm')`，然后 `naivegenerate(...)`
   - `bmeagle`：加载 `EaModel`，把每层 `layer.mlp` 换成 `SPMLP(..., forward_mode='bm')`，然后 `eagenerate(...)`
   - `deepspeed`：走 `deepspeed.init_inference(...)`
6. 对每条 prompt：tokenize、计时、生成、decode、累计平均时间 / AL / 峰值显存。

## 3. `SPMLP` 现在做了什么

`mlp.py` 里的 `SPMLP` 会先把 expert 权重重新打包，然后提供三条前向路径：

- `original_forward(...)`
  - 逐 expert 的 PyTorch 参考实现。
- `main_forward(...)`
  - 当 `batch_size * seq_len` 落在 `{32, 64, 128, 256}` 时，调用 `mlp_kernel.ops.sddmm(...)` 和 `mlp_kernel.ops.spmm(...)`。
  - 否则回退到 `original_forward(...)`。
- `bm_forward(...)`
  - 使用 `torch.bmm` 做对照实现。

当前 `SPMLP` 的结构假设是 Qwen3-MoE 风格：

- `origin_mlp.gate`
- `origin_mlp.experts`
- 每个 expert 都有 `gate_proj / up_proj / down_proj`
- `origin_mlp.top_k`
- `origin_mlp.num_experts`

这也是为什么它现在适配的是 Qwen3-MoE，而不是任意 MoE 模型。

## 4. `run.sh` 当前实际在做什么

仓库里的 `run.sh` 现在是：

```bash
set -e
CUDA_VISIBLE_DEVICES=6 \
python main.py --dataset 0 --method mtp
```

也就是说，默认实验是：

- 数据集：`alpaca`
- 方法：`mtp`
- 单卡运行：`CUDA_VISIBLE_DEVICES=6`

## 5. 2026-07-09 的实际复现实验

下面这些结论是我在当前机器上实际跑出来的，不是只看代码推断。

### 5.1 HF 冒烟测试

命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -s moe_test/main.py --dataset 0 --method hf --num_prompts 1
```

现象：

- 成功加载 `alpaca`
- 成功加载 `Qwen3-30B-A3B`
- 成功生成 1 条输出

记录到的结果：

- `Average time across 1 prompts: 19.1693 seconds`
- `Peak GPU memory usage: 57.09 GB`

### 5.2 EAGLE 冒烟测试

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -s moe_test/main.py --dataset 0 --method eagle --num_prompts 1
```

现象：

- 成功加载基座模型和 EAGLE 草稿模型
- 成功通过 `EaModel.eagenerate(...)` 跑完 1 条 prompt

记录到的结果：

- `Average time across 1 prompts: 47.9306 seconds`
- `Average AL (Acceptance Length) across 1 prompts: 2.7143`
- `Peak GPU memory usage: 28.75 GB`

这个结果只能当 smoke test，不能据此下最终速度结论。

### 5.3 MTP：为什么 `run.sh` 能跑，而我之前手动命令会报错

关键差异不是 conda 环境本身，而是运行方式：

- `run.sh` 是单卡：`CUDA_VISIBLE_DEVICES=6`
- 我之前复现失败时用的是多卡可见：`CUDA_VISIBLE_DEVICES=0,1`

当前机器上的实际结果如下。

#### 情况 A：按 `run.sh` 风格，单卡运行

命令：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
CUDA_VISIBLE_DEVICES=6 python -s main.py --dataset 0 --method mtp --num_prompts 1
```

结果：可以跑通。

一次实测输出为：

- `Average time across 1 prompts: 4.1836 seconds`
- `Average AL (Acceptance Length) across 1 prompts: 2.6111`
- `Peak GPU memory usage: 58.97 GB`

#### 情况 B：强制使用本仓库本地编译的 `mlp_kernel`，仍然单卡

命令：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
export PYTHONPATH=/share/zhouyongkang/projects/sc/moe_src:${PYTHONPATH}
CUDA_VISIBLE_DEVICES=6 python -s main.py --dataset 0 --method mtp --num_prompts 1
```

结果：也可以跑通。

一次实测输出为：

- `Average time across 1 prompts: 4.2214 seconds`
- `Average AL (Acceptance Length) across 1 prompts: 2.6111`
- `Peak GPU memory usage: 58.97 GB`

#### 情况 C：多卡可见

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -s moe_test/main.py --dataset 0 --method mtp --num_prompts 1
```

结果：当前仍可能触发：

```text
CUDA error: an illegal memory access was encountered
```

所以目前可以得出的结论是：

- `mtp` 在单卡 `run.sh` 风格路径上已经能跑。
- 多卡可见时，`mtp` 仍不稳定，需要单独排查。

## 6. `PYTHONPATH` 和导入路径差异

这个点很关键。

当前机器上，从 `moe_test/` 目录直接运行、又不显式设 `PYTHONPATH` 时，`import mlp_kernel` 可能会命中公共路径下的版本；而显式设置

```bash
export PYTHONPATH=/share/zhouyongkang/projects/sc/moe_src:${PYTHONPATH}
```

之后，才会强制使用当前仓库 `moe_src/mlp_kernel` 里刚编好的本地扩展。

因此要区分两件事：

- 想复现 `run.sh` 当前行为：先按默认方式跑。
- 想验证你在当前仓库里重编译出来的扩展：务必设 `PYTHONPATH`。

## 7. `main_old.py` 只当历史参考

`main_old.py` 是更早的原型，有几个明显问题：

- 交互式 `input(...)` 选方法，不适合自动化实验。
- 只覆盖 `hf` / `eagle` / `mtp`。
- 里面有一个明显的逻辑错误：

```python
elif generation_method == 'eagle' or 'mtp':
```

这个条件恒为真，所以不要把它当成当前可靠入口。

## 8. 当前最实用的结论

如果你现在要继续补实验，建议按下面理解 `moe_test`：

- 它本质上是 Qwen3 + EAGLE 的生成 benchmark harness。
- `hf` 和 `eagle` 路径已经能正常 smoke test。
- `mtp` 单卡路径已经能跑通。
- `mtp` 多卡路径仍可能出现 `illegal memory access`。
- 如果你要验证本地重新编译的 `mlp_kernel`，请显式设置 `PYTHONPATH`。

## 9. 2026-07-15 新增：根目录 uv 环境与 `run.sh` 实测命令

为了让 `moe_test/run.sh` 在当前仓库内自洽运行，我在项目根目录创建了一个 `uv` 虚拟环境：

- 路径：`/share/zhouyongkang/projects/sc/.venv`
- 方案：`uv venv --system-site-packages`，底层 Python 使用 `/share/zhouyongkang/conda_envs/mtpmoe_env/bin/python`
- 额外补装到 `.venv` 的包：`numpy==2.2.6`、`pytz`、`python-dateutil`、`tzdata`

同时我把 `run.sh` 改成了更稳的版本，现在它会：

- 自动优先使用根目录 `.venv/bin/python`
- 自动设置 `PYTHONNOUSERSITE=1`
- 自动把本仓库 `moe_src` 加入 `PYTHONPATH`
- 自动使用 `python -s`
- 支持把额外参数继续传给 `main.py`

因此现在不需要先 `source .venv/bin/activate`，直接运行 `bash run.sh ...` 即可。

### 9.1 我实际执行的命令

在项目根目录创建 uv 环境：

```bash
cd /share/zhouyongkang/projects/sc
uv venv --seed --system-site-packages --python /share/zhouyongkang/conda_envs/mtpmoe_env/bin/python .venv
uv pip install --python .venv/bin/python numpy==2.2.6 pytz python-dateutil tzdata
```

在 uv 环境里重编译本地 CUDA 扩展：

```bash
cd /share/zhouyongkang/projects/sc/moe_src
../.venv/bin/python -s setup.py build_ext --inplace --force
```

直接运行 `run.sh` 做 1 条 prompt 的冒烟验证：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
bash run.sh --num_prompts 1
```

如果你想显式指定卡，也可以写成：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
CUDA_VISIBLE_DEVICES=6 bash run.sh --num_prompts 1
```

### 9.2 实测结果

上面的 `bash run.sh --num_prompts 1` 已在 2026-07-15 跑通，输出末尾为：

```text
=== Results for MTP method on alpaca dataset ===
Average time across 1 prompts: 4.7576 seconds
Average AL (Acceptance Length) across 1 prompts: 2.6111
Peak GPU memory usage: 58.97 GB
```


## 10. `bm_fallback_t_d` 的实验计划

目标是为 `opt_mixer -> bm_forward` 的切换机制选一个合适的 `bm_fallback_t_d`。这里需要先区分两个量：

- `t_d_cfg`：算子执行时使用的配置值，也就是当前层真正按多少个 dense expert 去走主路径。
- `t_d_actual`：运行时根据路由结果统计出来的实际值，这里定义为“被选中次数严格大于 `maxnnz` 的专家个数”。

`bm_fallback_t_d` 要比较的不是理论复杂度，而是真实总开销，因此计时里要包含：

- `t_d_actual` 的统计开销
- 主路径进入 `sddmm/spmm` 前的整理开销
- `bm_forward_with_extra_input` 复用已有路由结果后的总开销

### 10.1 核心问题

需要回答两个问题：

1. 在固定超参数下，`t_d_actual` 增大到什么位置后，`bm` 路径开始稳定快于主路径？
2. 这个交点在不同层之间是否接近到可以共用一个全局阈值，还是必须按层设置？

我目前建议先做“控变量微基准”，再做“真实 prompt 分层统计”。

### 10.2 第一阶段：控变量微基准

目的：先看在固定 shape 下，两条路径的交点大致落在什么范围。

实验设置：

- `batch_size`：至少测 `64`、`128`
- `maxnnz`：至少测 `4`、`8`
- `t_d_cfg`：扫 `1..64`
- 其他超参数固定，并在实验记录中写明：
  - `hidden_size`
  - `intermediate_size`
  - `num_experts`
  - `top_k`
  - `sequence_length`
  - `dtype`
  - GPU 型号
  - CUDA / PyTorch 版本

计时对象：

- 主路径总时间：从拿到 `router_logits` 后开始，包含 `t_d_actual` 统计、稀疏整理、`sddmm`、`spmm`
- `bm` 路径总时间：从复用已有 `router_logits / routing_weights / selected_experts` 后开始，不重复做 gate / softmax / topk

判定方式：

- 对每个 `t_d_cfg` 重复多次，记录均值和方差
- 找到 `bm` 首次稳定不慢于主路径的位置，作为候选 `bm_fallback_t_d`
- “稳定”建议不是只看单点交叉，而是要求后续若干个 `t_d` 上也继续保持优势

### 10.3 第二阶段：真实 prompt 分层统计

目的：验证第一阶段得到的候选阈值，在真实推理流里是否仍成立，以及不同层是否一致。

建议配置：

- 数据：先用 `alpaca`
- prompt 数：先跑约 `30` 条
- 方法：`mtp`
- 每层、每次调用都记录：
  - `layer_id`
  - `batch_size`
  - `sequence_length`
  - `t_d_cfg`
  - `t_d_actual`
  - `route_prepare_ms`
  - `main_path_ms`
  - `bm_path_ms`
  - 若启用 fallback，记录最终选择的路径

重点分析：

- 同一个 `t_d_actual` 下，不同层的主路径 / `bm` 路径耗时是否接近
- 各层交点是否集中在一个小范围
- 某些层是否系统性更早或更晚适合切到 `bm`

### 10.4 阈值选择原则

如果实验结果满足下面条件，可以使用全局阈值：

- 大多数层的交点都落在很窄的范围内
- 用单一阈值带来的额外损失很小

如果层间差异明显，则改成按层阈值：

- 为每层单独统计交点
- 在模块初始化时加载 `layer_id -> bm_fallback_t_d` 映射

我当前的预期是：如果模型各层的 `hidden_size / intermediate_size / num_experts / top_k` 相同，交点大概率会比较接近，但不应直接假设“所有层完全相同”。原因是实际路由分布、专家命中模式、cache 行为和整理开销都可能让不同层出现偏移。

### 10.5 代码侧需要补的插桩

为了让这组实验可以直接落地，后续代码里建议补这些计时点：

- `mixer_with_fallback` 内：
  - `t_d_actual` 统计耗时
  - 稀疏整理耗时
- `main_forward` 内：
  - 主路径算子总耗时
- `bm_forward_with_extra_input` 内：
  - `bm` 计算总耗时
- `main.py` 或调用侧：
  - 汇总为按层 CSV / JSON，便于后处理画图

### 10.6 最终输出

这组实验最终建议产出三类结果：

- 全局交点图：横轴 `t_d_actual`，纵轴两条路径总耗时
- 分层交点图：每层一个交点或一个小提琴图 / 箱线图
- 阈值建议表：给出“全局阈值”或“逐层阈值”的最终配置建议

## 11. 2026-07-16 论文版 ablation 配置

这一轮 ablation 只重跑 `tech1 / tech2 / tech3`，`baseline` 直接复用 2026-07-15 已完成的 `eagle` 30-prompt 结果，不再重复计时。

### 11.1 baseline

复用的 baseline 定义：

- 方法：`eagle`
- 数据集：`alpaca`
- prompt 数：`30`
- 旧结果：`Average time across 30 prompts: 47.4339 seconds`
- 旧结果：`Average AL (Acceptance Length) across 30 prompts: 1.6710`

### 11.2 tech1 / tech2 / tech3 的新定义

为了和论文写法一致，这一轮把 `SPMLP.main_forward(...)` 里的 fallback 逻辑拆成了两类：

- unsupported-batch fallback
  - 当 `batch_size * sequence_length` 不在当前 CUDA 主路径支持的集合 `{32, 64, 128}` 时触发。
- runtime-td fallback
  - 当 `runtime_t_d > bm_fallback_t_d` 时触发。

本轮实验统一使用下面的定义：

- `tech1`
  - 方法：`mtp`
  - `t_d=64`
  - `adaptive_t_d=False`
  - unsupported-batch fallback：`original`
  - runtime-td fallback：`none`
- `tech2`
  - 方法：`mtp`
  - `t_d=64`
  - `adaptive_t_d=True`
  - unsupported-batch fallback：`original`
  - runtime-td fallback：`none`
- `tech3`
  - 方法：`mtp`
  - `t_d=64`
  - `adaptive_t_d=True`
  - unsupported-batch fallback：`original`
  - runtime-td fallback：`bm`
  - `bm_fallback_t_d=40`

换句话说：

- `tech1/tech2` 不再因为 unsupported shape 自动切到 `bm`，而是回到 `original_forward(...)`。
- 只有 `tech3` 在 `runtime_t_d` 足够大时才会切到 `bm_forward_with_extra_input(...)`。

### 11.3 为什么把 tech3 阈值先降到 40

目前仓库里已经有一份控变量微基准结果：

- 文件：`moe_test/bm_fallback_td_sweep.csv`
- 语义：在固定 shape、固定路由分布下，比对 `SPMLP` 主路径和 `bm fallback` 路径
- 注意：这份基准把“统计 `runtime_t_d` 的开销”和“fallback 前的 probe / mixer 开销”都计入了 `bm fallback` 路径

从这份已有结果可以直接读出：

- `(batch=64, maxnnz=4)`：`bm` 没有出现稳定胜出点
- `(batch=128, maxnnz=4)`：`bm` 没有出现稳定胜出点
- `(batch=64, maxnnz=8)`：`bm` 首次且稳定胜出从 `t_d=40` 开始
- `(batch=128, maxnnz=8)`：`bm` 首次且稳定胜出从 `t_d=19` 开始

而真实 30-prompt `runtime_t_d` 统计（见 `results/paper_numbers/runtime_td_summary.json`）又显示：

- `p90(runtime_t_d) = 33`
- `p95(runtime_t_d) = 36`
- `p99(runtime_t_d) = 42`
- `max(runtime_t_d) = 53`

所以：

- 之前的 `bm_fallback_t_d=56` 基本打不到，tech3 几乎不会触发 fallback。
- 这次先把阈值降到 `40`，目的是让 tech3 至少在真实推理里有一批可观测触发，而不是继续落在“几乎永不触发”的区间。

这不是最终定论，只是当前更适合做论文 ablation 的一个可观测配置。

### 11.4 关于“是否按层使用不同阈值”

目前还没有一份“逐层 main-vs-bm 真实总耗时交点”的完整结果，因此这一轮先不启用 `layer_id -> threshold` 的分层阈值表。

当前已有的数据只足够说明两件事：

- 不同层的 `runtime_t_d` 分布确实不同。
- 全局阈值 `56` 太高，几乎不会触发。

因此这轮先使用全局阈值 `40` 做一次 30-prompt ablation。若后续要认真做按层阈值，建议再补一轮“逐层总耗时交点”实验，而不是只根据 `runtime_t_d` 分布本身去设阈值。

### 11.5 本轮计划执行的命令

`baseline` 复用旧结果，不重跑。

`tech1`：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 \
PYTHONPATH=/share/zhouyongkang/projects/sc/moe_src:/share/zhouyongkang/projects/sc/moe_test \
/share/zhouyongkang/projects/sc/.venv/bin/python -s main.py \
  --dataset 0 --num_prompts 30 --method mtp \
  --spmlp-t-d 64 \
  --no-spmlp-adaptive-td \
  --spmlp-bm-fallback-td -1 \
  --spmlp-unsupported-fallback-mode original \
  --spmlp-runtime-fallback-mode none
```

`tech2`：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 \
PYTHONPATH=/share/zhouyongkang/projects/sc/moe_src:/share/zhouyongkang/projects/sc/moe_test \
/share/zhouyongkang/projects/sc/.venv/bin/python -s main.py \
  --dataset 0 --num_prompts 30 --method mtp \
  --spmlp-t-d 64 \
  --spmlp-adaptive-td \
  --spmlp-bm-fallback-td -1 \
  --spmlp-unsupported-fallback-mode original \
  --spmlp-runtime-fallback-mode none
```

`tech3`：

```bash
cd /share/zhouyongkang/projects/sc/moe_test
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 \
PYTHONPATH=/share/zhouyongkang/projects/sc/moe_src:/share/zhouyongkang/projects/sc/moe_test \
/share/zhouyongkang/projects/sc/.venv/bin/python -s main.py \
  --dataset 0 --num_prompts 30 --method mtp \
  --spmlp-t-d 64 \
  --spmlp-adaptive-td \
  --spmlp-bm-fallback-td 40 \
  --spmlp-unsupported-fallback-mode original \
  --spmlp-runtime-fallback-mode bm
```

### 11.6 新增插桩

为了便于解释结果，本轮 `SPMLP` 还会额外汇总并打印下面这些计数：

- `kernel_path_calls`
- `unsupported_batch_fallback_calls`
- `runtime_t_d_fallback_calls`
- `original_forward_calls`
- `bm_forward_calls`

这些计数可以帮助判断：

- `tech1/tech2` 里有多少调用实际上没有进入 CUDA 主路径
- `tech3` 里 `bm` fallback 到底触发了多少次
- 端到端差异为什么会比单层微基准更小

### 11.7 2026-07-16 实测结果

这三组都使用：

- 数据集：`alpaca`
- prompt 数：`30`
- GPU：`CUDA_VISIBLE_DEVICES=6`
- Python：`/share/zhouyongkang/projects/sc/.venv/bin/python -s`
- `PYTHONPATH=/share/zhouyongkang/projects/sc/moe_src:/share/zhouyongkang/projects/sc/moe_test`

复用的 `baseline`：

- `Average time across 30 prompts: 47.4339 seconds`
- `Average AL (Acceptance Length) across 30 prompts: 1.6710`

本轮新跑结果：

- `tech1`
  - `Average time across 30 prompts: 5.3035 seconds`
  - `Average AL (Acceptance Length) across 30 prompts: 1.5916`
  - `SPMLP path stats: layers=48 kernel_path_calls=73536 unsupported_batch_fallback_calls=1440 runtime_t_d_fallback_calls=0 original_forward_calls=1440 bm_forward_calls=0`
- `tech2`
  - `Average time across 30 prompts: 4.7313 seconds`
  - `Average AL (Acceptance Length) across 30 prompts: 1.6343`
  - `SPMLP path stats: layers=48 kernel_path_calls=72432 unsupported_batch_fallback_calls=1440 runtime_t_d_fallback_calls=0 original_forward_calls=1440 bm_forward_calls=0`
- `tech3`
  - `Average time across 30 prompts: 4.6757 seconds`
  - `Average AL (Acceptance Length) across 30 prompts: 1.5991`
  - `SPMLP path stats: layers=48 kernel_path_calls=72548 unsupported_batch_fallback_calls=1440 runtime_t_d_fallback_calls=1084 original_forward_calls=1440 bm_forward_calls=1084`

对 `baseline` 的时间优化量：

- `tech1`
  - 节省 `42.1304s`
  - 端到端时间下降 `88.82%`
  - 相对 `baseline` 加速 `8.94x`
- `tech2`
  - 节省 `42.7026s`
  - 端到端时间下降 `90.03%`
  - 相对 `baseline` 加速 `10.03x`
- `tech3`
  - 节省 `42.7582s`
  - 端到端时间下降 `90.14%`
  - 相对 `baseline` 加速 `10.14x`

技术点之间的增量效果：

- `tech2` 相对 `tech1`
  - 再快 `0.5722s`
  - 再降时 `10.79%`
- `tech3` 相对 `tech2`
  - 再快 `0.0556s`
  - 再降时 `1.18%`
- `tech3` 相对 `tech1`
  - 再快 `0.6278s`
  - 再降时 `11.84%`

当前这轮结果说明：

- 把 `unsupported-batch` fallback 改回 `original` 之后，`tech1` 明显慢于之前那版使用 `bm` fallback 的结果。
- `adaptive_t_d` 本身仍然有清晰收益，`tech2` 相对 `tech1` 继续提升。
- `tech3` 在 `bm_fallback_t_d=40` 时确实会触发大量 `bm` fallback（1084 次），但相对 `tech2` 的额外收益只有约 `1.18%`，说明在当前 `maxnnz=4` 设定下，`bm` 分支的边际收益仍然比较有限。

## 12. 2026-07-16 固定 `bs=128, maxnnz=4` 的控变量 ablation

这组实验的目的是回答一个更具体的问题：

- 在统一 `batch_size=128`、`sequence_length=1`、`maxnnz=4` 的情况下，`tech2` 相对 `tech1` 是否能达到平均 `1.1x` 提升？
- 在同样条件下，`tech3` 相对 `tech2` 是否也能达到平均 `1.1x` 提升？

这里不再测 `baseline`，只比较 `tech1 / tech2 / tech3` 三个技术点本身。

### 12.1 定义

统一配置：

- `batch_size = 128`
- `sequence_length = 1`
- `tokens = 128`
- `maxnnz = 4`
- `hidden_size = 2048`
- `intermediate_size = 768`
- `num_experts = 128`
- `top_k = 8`
- `t_d_cfg = 64`

方法定义：

- `tech1`
  - `adaptive_t_d = False`
  - `runtime fallback = none`
- `tech2`
  - `adaptive_t_d = True`
  - `runtime fallback = none`
- `tech3`
  - `adaptive_t_d = True`
  - `runtime fallback = bm`
  - 额外扫 `bm_fallback_t_d = 0..64`

### 12.2 平均方式

这组实验不是直接跑真实 prompt，而是构造一系列可达的 `runtime_t_d`，然后在这些点上做统一平均。

对 `bs=128,maxnnz=4`，可达的 `runtime_t_d` 为：

`[5, 6, 7, ..., 64]`

最终平均采用：

- 对所有可达 `runtime_t_d` 做等权平均
- 同时报告：
  - `ratio of means`
  - `geometric mean of per-t_d speedups`
  - `arithmetic mean of per-t_d speedups`

### 12.3 结果

结果文件：

- `results/bs128_maxnnz4_ablation/bs128_maxnnz4_ablation.csv`
- `results/bs128_maxnnz4_ablation/bs128_maxnnz4_ablation_summary.json`
- `results/bs128_maxnnz4_ablation/bs128_maxnnz4_ablation.md`

`tech2` 相对 `tech1`：

- `mean latency tech1 = 1.372242 ms`
- `mean latency tech2 = 0.913496 ms`
- `ratio of means = 1.502198x`
- `geometric mean speedup = 1.576755x`
- `arithmetic mean speedup = 1.706223x`

结论：

- `tech2` 相对 `tech1` 可以达到并明显超过平均 `1.1x` 提升。

`tech3` 相对 `tech2`，若沿用之前在真实推理里试过的 `bm_fallback_t_d = 40`：

- `mean latency tech3 = 0.997315 ms`
- `ratio of means = 0.915949x`
- `geometric mean speedup = 0.935642x`
- `arithmetic mean speedup = 0.949439x`

这说明：

- 在这个固定配置下，`tech3` 不但达不到平均 `1.1x`，反而比 `tech2` 更慢。

### 12.4 是否存在某个更好的阈值

这轮还把 `bm_fallback_t_d = 0..64` 全部扫了一遍。

最优结果是：

- `best ratio-of-means threshold = 64`
- 对应 `tech3 / tech2 = 1.0x`

也就是说：

- 在 `bs=128,maxnnz=4` 这个设定下，最优策略其实是“完全不要触发 bm fallback”。
- 一旦真的让 `tech3` 在某些 `runtime_t_d` 上切到 `bm`，平均性能只会下降。

### 12.5 结论

所以如果论文里想在 `bs=128,maxnnz=4` 这个统一设定下同时宣称：

- `tech2` 相对 `tech1` 平均 `>= 1.1x`
- `tech3` 相对 `tech2` 平均 `>= 1.1x`

那么当前实现和当前硬件上，这个结论不成立。

更准确的说法应当是：

- `tech2` 在该设定下成立，而且裕量很大。
- `tech3` 在该设定下不成立；最优阈值实际上退化为“不做 bm fallback”。

## 13. 2026-07-16 固定 `bs=64, maxnnz=8` 的控变量 ablation

这组实验沿用上一节完全相同的方法，只把配置改成：

- `batch_size = 64`
- `sequence_length = 1`
- `maxnnz = 8`

其余保持不变：

- `hidden_size = 2048`
- `intermediate_size = 768`
- `num_experts = 128`
- `top_k = 8`
- `t_d_cfg = 64`

定义仍然是：

- `tech1`：`adaptive_t_d=False`，`runtime fallback=none`
- `tech2`：`adaptive_t_d=True`，`runtime fallback=none`
- `tech3`：`adaptive_t_d=True`，`runtime fallback=bm`，并对 `bm_fallback_t_d=0..64` 扫描

### 13.1 平均方式

在 `bs=64,maxnnz=8` 下，可达的 `runtime_t_d` 为：

`[1, 2, 3, ..., 56]`

最终平均采用：

- 对所有可达 `runtime_t_d` 做等权平均
- 同时报告：
  - `ratio of means`
  - `geometric mean of per-t_d speedups`
  - `arithmetic mean of per-t_d speedups`

### 13.2 结果文件

- `results/bs64_maxnnz8_ablation/bs64_maxnnz8_ablation.csv`
- `results/bs64_maxnnz8_ablation/bs64_maxnnz8_ablation_summary.json`
- `results/bs64_maxnnz8_ablation/bs64_maxnnz8_ablation.md`

### 13.3 结果

`tech2` 相对 `tech1`：

- `mean latency tech1 = 1.701883 ms`
- `mean latency tech2 = 0.987423 ms`
- `ratio of means = 1.723570x`
- `geometric mean speedup = 1.826761x`
- `arithmetic mean speedup = 1.933001x`

结论：

- `tech2` 相对 `tech1` 明显超过平均 `1.1x` 提升。

`tech3` 相对 `tech2`，若使用最优阈值搜索结果：

- `best threshold = 40`
- `mean latency tech3 = 0.946700 ms`
- `ratio of means = 1.043010x`
- `geometric mean speedup = 1.030882x`
- `arithmetic mean speedup = 1.034778x`

这说明：

- 在 `bs=64,maxnnz=8` 这个设定下，`tech3` 相对 `tech2` 的平均收益是正的。
- 但它仍然达不到你想要的平均 `1.1x`。

### 13.4 结论

因此在 `bs=64,maxnnz=8` 这组控变量实验中：

- `tech2 > tech1`：成立，而且裕量很大。
- `tech3 > tech2`：成立，但只有约 `1.043x`，不足以支撑“平均 `1.1x` 提升”的写法。

## 14. 2026-07-16 `alpaca` 真实 30-prompt：`ea_total_token=128, maxnnz=8`

前面第 12、13 节都是控变量 micro-benchmark，会人为构造固定 `batch_size / runtime_t_d`。你后来要求这组不要再“手工改 `runtime_t_d`”，而是直接在真实数据集上跑，因此这里单独补了一组真实生成实验。

### 14.1 实验设置

- 数据集：`alpaca`
- prompt 数：`30`
- 生成入口：`moe_test/main.py`
- 最大生成长度：`max_new_tokens=128`
- “bs128” 的对应设定：`ea_total_token=128`
- `maxnnz=8`
- `t_d_cfg=64`
- `tech3` 阈值：`bm_fallback_t_d=19`

这里的“bs128”要特别说明一下：

- 这组实验不是像第 12、13 节那样，把每次 MLP 调用强行固定成 `batch_size=128, sequence_length=1`。
- 这里跑的是 EAGLE/MTP 的真实推理过程，因此一次完整生成里会混合出现多种 shape。
- 我们这里只是把 `ea_total_token` 设成 `128`，让它对应你想看的 `bs128` 配置。

`tech3` 先取 `19`，原因是前面控变量 `bm_fallback_t_d` sweep 里，`(batch=128, maxnnz=8)` 的主路径和 `bm` 路径第一次稳定交叉点就在 `t_d=19` 左右。

### 14.2 复现脚本和结果文件

我新增了脚本：

- `moe_test/run_real_alpaca_bs128_maxnnz8.py`

它会顺序运行：

- `tech1`
- `tech2`
- `tech3`
- `bm`
- `bmeagle`

并把结果保存到：

- `results/real_alpaca_bs128_maxnnz8/summary.csv`
- `results/real_alpaca_bs128_maxnnz8/summary.json`
- `results/real_alpaca_bs128_maxnnz8/summary.md`
- `results/real_alpaca_bs128_maxnnz8/logs/*.log`

### 14.3 真实结果

平均每个 prompt 的耗时如下：

- `tech1`：`10.6905 s`
- `tech2`：`6.9731 s`
- `tech3`：`5.6641 s`
- `bm`：`7.5346 s`
- `bmeagle`：`4.8210 s`

对应 Acceptance Length：

- `tech1`：`1.7068`
- `tech2`：`1.6605`
- `tech3`：`1.7391`
- `bmeagle`：`1.6755`

`bm` 走的是 `naivegenerate(...)`，这里没有 AL 指标。

### 14.4 相对速度

从这组真实 `alpaca` 30-prompt 结果直接算：

- `tech2 / tech1 = 1.5331x`
- `tech3 / tech2 = 1.2311x`
- `tech3 / bm = 1.3302x`
- `bmeagle / tech3 = 1.1749x`

因此在这组真实实验里：

- `tech2` 相对 `tech1` 明显成立，而且超过 `1.1x`
- `tech3` 相对 `tech2` 也成立，而且这次超过了 `1.1x`
- 纯 `bm` 路径不如 `tech3`
- `bmeagle` 仍然是这组里最快的

### 14.5 路径统计怎么解释

`summary.csv` 里对应的 `SPMLP` 路径统计是：

- `tech1`
  - `kernel_path_calls=70608`
  - `unsupported_batch_fallback_calls=1440`
  - `runtime_t_d_fallback_calls=0`
  - `original_forward_calls=1440`
  - `bm_forward_calls=0`
- `tech2`
  - `kernel_path_calls=71904`
  - `unsupported_batch_fallback_calls=1440`
  - `runtime_t_d_fallback_calls=0`
  - `original_forward_calls=1440`
  - `bm_forward_calls=0`
- `tech3`
  - `kernel_path_calls=3051`
  - `unsupported_batch_fallback_calls=1440`
  - `runtime_t_d_fallback_calls=66981`
  - `original_forward_calls=1440`
  - `bm_forward_calls=66981`
- `bm`
  - `bm_forward_calls=187200`
- `bmeagle`
  - `bm_forward_calls=72864`

这说明：

- `tech1 -> tech2` 的收益主要来自自适应 `t_d`，而不是 `bm`。
- 在 `ea_total_token=128,maxnnz=8` 这个真实配置下，`runtime_t_d > 19` 的情况非常多，所以 `tech3` 会大量触发 `bm` fallback。
- 这些 fallback 在这次真实实验里不是负担，而是净收益，所以 `tech3` 比 `tech2` 继续快了约 `1.23x`。
- 但如果只看绝对最快，当前还是 `bmeagle` 更快。

