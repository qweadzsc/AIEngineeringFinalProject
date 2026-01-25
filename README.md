## Environment Preparation

### Hardware Requirements
The operator implementation in this article uses NVIDIA CUTLASS and leverages Ampere architecture features, so it is best tested on NVIDIA hardware with Ampere architecture or newer. The hardware used for testing in this article is NVIDIA A800 with CUDA version 12.2.

### CUTLASS
Refer to the official CUTLASS repository for downloading and compiling CUTLASS: https://github.com/NVIDIA/cutlass

### Python
After completing the CUTLASS compilation, modify the `cutlass_path` in `moe_src/setup.py` to point to the CUTLASS directory mentioned above, then run the following commands to create a conda environment and install the operators provided by this method:

```bash
conda create -n sd python=3.10 -y
conda activate sd
pip install -r requirement.txt
cd moe_src
python setup.py install
```

### Models and Datasets
The datasets used in this article are also included in the repository, but the models need to be downloaded separately. The test model used in this article is [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), and the draft model used is the [EAGLE3 model](https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3). After downloading the models, you need to modify the corresponding paths in `moe_test/main.py` to the actual model locations.


## Performance Testing

Use the following command to run efficiency tests:

```bash
python moe_test/main.py --dataset <dataset_id> --method <impl_name> --num_prompts <n>
```

Where:

- The `dataset` parameter specifies the test dataset to use, which can be set to 0-7, corresponding to the datasets: alpaca, commonsense_qa, gsm8k, hellaswag, piqa, siqa, sst2, sum respectively

- The `method` parameter specifies the implementation method, with options: `hf`, `eagle`, `mtp`, `deepspeed`. Among these, `hf` represents the baseline implementation using HuggingFace and PyTorch, `eagle` is the directly fused baseline (applying the EAGLE method directly on the MoE model), `mtp` is our proposed method, and `deepspeed` is another baseline using DeepSpeed's MoE optimization method

- The `num_prompts` parameter specifies the number of test samples to use; if not specified, all data will be used by default.
