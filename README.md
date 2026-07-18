# 项目环境配置与迁移说明

这份文档面向“把本项目迁移到另一台机器继续补实验”的场景，重点说明如何在新硬件上完成环境准备、路径修改、CUDA 扩展编译和最小验证。

## 1. 项目结构

项目主要分成两部分：

- `moe_src`：自定义 CUDA / Triton 算子、单元测试、性能对比脚本。
- `moe_test`：基于 Qwen3-30B-A3B 和 EAGLE 草稿模型的推理实验入口。

如果你只补算子性能实验，重点看 `moe_src` 即可；如果你要补端到端推理实验，还需要准备 `moe_test` 的模型环境。

## 2. 硬件要求

### 2.1 GPU

推荐使用 NVIDIA Ampere 或更新架构的 GPU。本项目当前 CUDA 扩展默认按 `sm_80` 编译，也就是面向 A100 / A800 这类卡。

如果新机器不是 `sm_80`，需要手动改架构参数，否则可能出现：

- 编译通过但性能异常
- 编译失败
- 运行时报错

### 2.2 显存

端到端推理实验对显存要求较高：

- `hf` 方法通常显存占用最高。
- `eagle` / `mtp` 是否需要多卡，取决于模型装载方式和单卡显存；当前机器上 `mtp` 在单张 80GB 卡上可以跑通，`eagle` 则用过 2 卡做验证。
- 只跑 `moe_src` 的算子测试时，显存压力远小于完整模型推理。

### 2.3 系统环境

建议：

- Linux
- Python 3.10
- 已正确安装 NVIDIA 驱动
- 已安装 CUDA Toolkit
- `gcc/g++`
- `ninja`

## 3. 推荐的软件环境

本项目根目录提供了 `requirements.txt`，建议先创建一个新的 conda 环境：

```bash
conda create -n sc_moe python=3.10 -y
conda activate sc_moe
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

说明：

- `requirements.txt` 中已经包含 `torch`、`transformers`、`deepspeed`、`datasets`、`matplotlib`、`triton` 等依赖。
- 如果你的机器上已经有一套必须保留的 PyTorch / CUDA 组合，也可以先手动安装与你机器匹配的 PyTorch，再补装其他依赖。
- 最重要的一点是：`moe_src` 的 CUDA 扩展必须在“最终运行 `mtp` / 测试脚本的同一个 Python 环境里”重新编译，不能混用别的环境里编出来的 `.so` 文件。
- 之前用于验证 `mtp` 的临时环境 `.tmp_sc_moe_mtp_clone` 已经删除；后续迁移请在你自己正式使用的 conda 环境里完成编译和运行。

## 4. CUTLASS 准备

项目依赖 CUTLASS 头文件。默认代码假设 CUTLASS 位于：

```bash
deps/cutlass
```

也就是期望存在：

```bash
deps/cutlass/include
```

如果新机器上没有这个目录，可以自行准备 CUTLASS，并把路径改到 `moe_src/setup.py` 里的 `cutlass_path`：

```python
cutlass_path = "/你的路径/cutlass"
```

当前代码里这个变量在：

- `moe_src/setup.py`

## 5. 迁移到新 GPU 时必须检查的两个架构参数

### 5.1 `moe_src/setup.py`

当前默认写死为：

```python
'-arch=sm_80'
```

如果新机器不是 `sm_80`，需要改成对应架构。

### 5.2 `moe_src/run_test.sh`

当前默认写的是：

```bash
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
```

如果新机器不是 8.0，也要一起改掉，或者直接在运行前通过环境变量覆盖。

### 5.3 如何查询当前 GPU 的 capability

可以用下面的命令查询：

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

例如如果输出是：

```bash
('NVIDIA A800', (8, 0))
```

那么：

- `setup.py` 里应使用 `sm_80`
- `TORCH_CUDA_ARCH_LIST` 应使用 `8.0`

## 6. 必须修改的硬编码路径

迁移到新机器时，下面这些地方通常都要改。

### 6.1 模型路径

`moe_test/main.py` 里当前写死了：

- `base_model_path = "/share/public/public_models/Qwen3-30B-A3B"`
- `EAGLE_model_path = "/share/zhouyongkang/models/qwen3_30b_moe_eagle3"`

新机器上请改成你自己的模型目录。

### 6.2 CUTLASS 路径

`moe_src/setup.py` 中：

- `cutlass_path = "/share/zhouyongkang/projects/sc/deps/cutlass"`

如果仓库位置变了，或者 CUTLASS 不放在仓库内，需要同步修改。

### 6.3 默认 GPU 编号

下面这些脚本有默认 `CUDA_VISIBLE_DEVICES` 配置：

- `moe_src/run_test.sh`
- `moe_test/run.sh`

例如 `moe_test/run.sh` 当前默认是卡 `6`。迁移到新机器后，建议不要依赖脚本里的默认值，直接在命令行显式指定。

### 6.4 其他非主流程脚本

以下文件也有绝对路径，但不是主实验的核心入口；只有在你确实要跑这些脚本时才需要改：

- `moe_test/main_old.py`
- `moe_src/test/reorder.py`

## 7. 编译本地 CUDA 扩展

这是迁移过程中最关键的一步。

进入 `moe_src` 目录后，使用当前环境重新编译：

```bash
cd /你的项目路径/sc/moe_src
python setup.py build_ext --inplace --force
```

建议使用 `build_ext --inplace --force`，而不是只依赖以前装进 `site-packages` 的版本。原因是：

- 本地扩展 `mlp_kernel._C.abi3.so` 和当前 PyTorch 版本存在 ABI 绑定关系。
- 如果换了机器、换了 Python 环境、换了 PyTorch 版本，却继续使用旧 `.so`，很容易出现 `undefined symbol` 之类的导入错误。

## 8. 运行时的 `PYTHONPATH` 设置

如果你要验证“当前仓库里刚编好的本地扩展”，建议运行推理实验前显式设置：

```bash
export PYTHONPATH=/你的项目路径/sc/moe_src:${PYTHONPATH}
```

如果不这样做，Python 有可能优先导入系统里别的位置上的 `mlp_kernel`，导致：

- 导入到错误版本的 `.so`
- 新编好的扩展没有真正被使用
- 出现 ABI 不匹配

当前机器上已经观察到一种典型情况：

- 在 `moe_test/` 目录直接跑 `python main.py ...`，如果不设 `PYTHONPATH`，可能会导入公共路径下的 `mlp_kernel`。
- 显式设置 `PYTHONPATH=/你的项目路径/sc/moe_src:${PYTHONPATH}` 之后，才会强制使用当前仓库本地重新编译的扩展。

因此这里有两个用途需要区分：

- 想复现 `moe_test/run.sh` 当前行为：可以先按脚本默认方式运行。
- 想验证你刚在本仓库里重新编译的 `mlp_kernel`：务必显式设置 `PYTHONPATH`。

## 9. 数据和模型准备

### 9.1 数据集

仓库里已经包含了 `benchmark/` 目录下的大部分测试数据，因此通常不需要额外下载数据集。

### 9.2 模型

端到端实验需要至少准备两个模型：

- 基座模型：`Qwen3-30B-A3B`
- 草稿模型：`qwen3_30b_moe_eagle3`

准备好之后，把它们的本地路径填到：

- `moe_test/main.py`

如果你只跑 `moe_src` 的算子性能实验，则不需要下载这些大模型。

## 10. 最小验证流程

建议按下面顺序验证，而不是一上来直接跑 `mtp`。

### 10.1 先验证 CUDA 和 PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 10.2 再编译扩展

```bash
cd /你的项目路径/sc/moe_src
python setup.py build_ext --inplace --force
```

### 10.3 跑 `moe_src` 的基础测试

```bash
cd /你的项目路径/sc/moe_src
bash run_test.sh
```

如果只想先跑最基础的一项：

```bash
cd /你的项目路径/sc/moe_src
python -m unittest -v test.test_mixed_sddmm
```

### 10.4 跑 `moe_test` 的推理冒烟测试

先进入项目根目录。如果你要验证本地刚编好的扩展，再额外设置：

```bash
cd /你的项目路径/sc
export PYTHONPATH=/你的项目路径/sc/moe_src:${PYTHONPATH}
```

然后建议按这个顺序测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -s moe_test/main.py --dataset 0 --method hf --num_prompts 1
CUDA_VISIBLE_DEVICES=0,1 python -s moe_test/main.py --dataset 0 --method eagle --num_prompts 1
CUDA_VISIBLE_DEVICES=6 python -s moe_test/main.py --dataset 0 --method mtp --num_prompts 1
```

含义如下：

- `hf`：纯 HuggingFace 基线
- `eagle`：EAGLE 推理
- `mtp`：自定义 MoE MLP 路径
- `bm`：`torch.bmm` 基线
- `bmeagle`：EAGLE + `torch.bmm` 基线

如果你要补对照实验，`bm` / `bmeagle` 也建议一起验证。

关于 `mtp`，当前机器上的实际现象要单独说明：

- `moe_test/run.sh` 当前等价于在 `moe_test/` 目录下执行单卡 `CUDA_VISIBLE_DEVICES=6 python main.py --dataset 0 --method mtp`。
- 这条单卡路径已经实测可以跑通。
- 但把同样的 `mtp` 命令改成多卡可见，例如 `CUDA_VISIBLE_DEVICES=0,1`，仍可能触发 `CUDA error: an illegal memory access was encountered`。

所以迁移到新机器后，建议先用单卡把 `mtp` 跑通，再去排查多卡场景。

## 11. `moe_src` 常用实验命令

### 11.1 主 CUDA kernel 与 `bmm` 对比图

```bash
cd /你的项目路径/sc
RUN_TORCH_CUDA_BENCH=1 python -m unittest -v moe_src.test.test_torch_cuda
```

输出文件默认保存在：

- `moe_src/test/torch_cuda_td_sweep.png`
- `moe_src/test/torch_cuda_td_sweep.csv`
- `moe_src/test/torch_cuda_td_sweep_metadata.txt`

### 11.2 hybrid 路径与 `bmm` 对比图

```bash
cd /你的项目路径/sc
RUN_TORCH_CUDA_HYBRID_BENCH=1 python -m unittest -v moe_src.test.test_torch_cuda_hybrid
```

输出文件默认保存在：

- `moe_src/test/torch_cuda_hybrid_td_sweep.png`
- `moe_src/test/torch_cuda_hybrid_td_sweep.csv`
- `moe_src/test/torch_cuda_hybrid_td_sweep_metadata.txt`

## 12. 常见问题

### 12.1 `ImportError: ... undefined symbol ...`

这通常说明 `mlp_kernel` 是在别的 Python / PyTorch 环境里编出来的，和当前环境 ABI 不兼容。

处理方式：

1. 进入当前要运行实验的 conda 环境。
2. 删除对旧安装版本的依赖。
3. 在 `moe_src` 目录重新执行：

```bash
python setup.py build_ext --inplace --force
```

4. 运行前显式设置：

```bash
export PYTHONPATH=/你的项目路径/sc/moe_src:${PYTHONPATH}
```

### 12.2 `CUDA error: an illegal memory access was encountered`

当前项目里的 `mtp` 路径在某些情况下可能出现异步 CUDA 报错。当前机器上已经确认：

- 单卡 `run.sh` 风格命令可以跑通。
- 多卡可见时同一条 `mtp` 命令仍可能报 `illegal memory access`。

迁移到新机器后如果遇到这个问题，建议先这样定位：

```bash
CUDA_LAUNCH_BLOCKING=1 python -s moe_test/main.py --dataset 0 --method mtp --num_prompts 1
```

这条命令的作用不是“永久修复”，而是为了把异步报错变成更容易定位的同步报错。

### 12.3 `transformers` / `datasets` / `deepspeed` 缺失

说明当前环境不是完整推理环境，重新检查：

```bash
pip install -r requirements.txt
```

### 12.4 找不到模型或数据路径

请优先检查：

- `moe_test/main.py` 里的模型绝对路径
- 当前工作目录是否是项目根目录
- `benchmark/` 是否完整存在

## 13. 迁移时的推荐顺序

建议严格按下面顺序做：

1. 把仓库复制到新机器。
2. 准备 Python 3.10 环境并安装 `requirements.txt`。
3. 准备 CUTLASS，并修改 `moe_src/setup.py` 的 `cutlass_path`。
4. 查询新 GPU 的 compute capability，并同步修改 `sm_80` / `TORCH_CUDA_ARCH_LIST=8.0`。
5. 修改 `moe_test/main.py` 中的模型路径。
6. 在新环境里重新执行 `python setup.py build_ext --inplace --force`。
7. 如果你要验证本地扩展，设置 `PYTHONPATH=/你的项目路径/sc/moe_src:${PYTHONPATH}`。
8. 先跑 `moe_src` 的基础测试，再跑 `hf` / `eagle` / `mtp` 冒烟测试。
9. 最后再开始正式 benchmark。

## 14. 子目录文档

更具体的实验说明分别在：

- `moe_src/README.md`
- `moe_test/README.md`
- `GPT_OSS_MIGRATION.md`

根目录这份文档主要解决“迁移到新机器时怎样把环境重新配起来”，细节 benchmark 解释请看各自子目录文档。
