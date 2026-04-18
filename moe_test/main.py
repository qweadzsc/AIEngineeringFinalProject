import torch
import time
import argparse
import os
# Assuming data.py is in the same directory or in the Python path
from data import CustomTextDataset, resolve_dataset_path
from tqdm import tqdm
from EAGLE.eagle.model.ea_model import EaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlp import SPMLP
import deepspeed


def run_hf_generation(model, tokenizer, input_ids, max_new_tokens, temperature=0.0, top_p=None, top_k=None):
    """Run generation using standard Hugging Face implementation."""
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0.0 or (top_p is not None and top_p < 1.0) or (top_k is not None and top_k > 0):
        generate_kwargs["do_sample"] = True
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if top_k is not None:
            generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs["do_sample"] = False
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    return outputs


def run_eagle_generation(model, input_ids, max_new_tokens):
    """Run generation using EAGLE implementation."""
    output_ids, al = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
    return output_ids, al


def run_bm_generation(model, input_ids, max_new_tokens):
    """Run generation using EAGLE implementation."""
    output_ids = model.naivegenerate(input_ids, max_new_tokens=max_new_tokens)
    return output_ids


def run_deepspeed_generation(model, tokenizer, input_ids, max_new_tokens, temperature=0.0, top_p=None, top_k=None):
    """Run generation using DeepSpeed inference engine."""
    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0.0 or (top_p is not None and top_p < 1.0) or (top_k is not None and top_k > 0):
        generate_kwargs["do_sample"] = True
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if top_k is not None:
            generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs["do_sample"] = False
    
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    return outputs


def load_deepspeed_model(base_model_path, max_tokens=128):
    """Load model with DeepSpeed inference optimizations."""
    
    print(f"Loading model for DeepSpeed inference from {base_model_path}...")
    
    # Load the base model first
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    
    # 配置DeepSpeed推理 - 使用正确的推理配置格式
    ds_config = {
        "dtype": torch.float16,  # 数据类型
        "replace_method": "auto",  # 自动替换优化层
        "replace_with_kernel_inject": True,  # 使用内核注入优化
        "enable_cuda_graph": False,  # 是否启用CUDA图（根据模型支持情况）
    }
    
    # 如果模型支持MoE（Mixture of Experts），可能需要特殊处理
    if "moe" in base_model_path.lower() or hasattr(model.config, "num_experts"):
        ds_config["replace_with_kernel_inject"] = False
        print("Warning: MoE models may not support kernel injection. Disabling it.")
    
    # Initialize DeepSpeed inference
    ds_engine = deepspeed.init_inference(
        model=model,
        config=ds_config
    )
    
    # Get the optimized model
    model = ds_engine.module
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Run LLM generation with different methods.")
    parser.add_argument("--dataset", type=int, default=7, 
                        help="Index of the dataset in ['alpaca', 'commonsense_qa', 'gsm8k', 'hellaswag', 'piqa', 'siqa', 'sst2', 'sum'] (default: 7)")
    parser.add_argument("--method", type=str, default='hf', choices=['hf', 'eagle', 'mtp', 'deepspeed', 'bm', 'bmeagle'],
                        help="Generation method: hf (HuggingFace), eagle, mtp, bm, bmeagle, or deepspeed (default: hf)")
    parser.add_argument("--num_prompts", type=int, default=10,
                        help="Number of prompts to process (default: all)")
    args = parser.parse_args()

    datasets_names = ['alpaca', 'commonsense_qa', 'gsm8k', 'hellaswag', 'piqa', 'siqa', 'sst2', 'sum']
    
    if args.dataset < 0 or args.dataset >= len(datasets_names):
        raise ValueError(f"Dataset index {args.dataset} is out of range for list of {len(datasets_names)} datasets.")
        
    dataset_name = datasets_names[args.dataset]
    dataset_path = resolve_dataset_path(f'benchmark/{dataset_name}')
    print(f"Loading dataset: {dataset_name} from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist. Attempting to load anyway...")
    
    dataset = CustomTextDataset(dataset_path)

    base_model_path = "/share/public/public_models/Qwen3-30B-A3B"
    EAGLE_model_path = "/share/zhouyongkang/models/qwen3_30b_moe_eagle3"

    generation_method = args.method.lower()
    print(f"Selected generation method: {generation_method}")

    model = None
    tokenizer = None

    if generation_method == 'hf':
        print(f"Loading standard Hugging Face model from {base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif generation_method == 'eagle':
        print(f"Loading EAGLE model with base {base_model_path} and EA model {EAGLE_model_path}...")
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=EAGLE_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=64,
            # use_eagle3=False,
        )
        model.eval()
        model.device = model.base_model.device
        tokenizer = model.tokenizer

    elif generation_method == 'mtp' or generation_method == 'bm' or generation_method == "bmeagle":
        print(f"Loading EAGLE model with base {base_model_path} and EA model {EAGLE_model_path} for MTP...")
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=EAGLE_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=64,
            # use_eagle3=False,
        )
        model.eval()
        model.device = model.base_model.device
        forward_mode = "bm" if generation_method in {"bm", "bmeagle"} else "main"
        for layer in tqdm(model.base_model.model.layers, desc=f"Applying SPMLP ({forward_mode}) to layers"):
            layer.mlp = SPMLP(layer.mlp, forward_mode=forward_mode)
        tokenizer = model.tokenizer

    elif generation_method == 'deepspeed':
        model, tokenizer = load_deepspeed_model(base_model_path)

    else:
        raise NotImplementedError(f"Unknown method {generation_method}")

    total_time = 0
    total_al_mean = 0
    num_prompts = min(args.num_prompts, len(dataset))
    print(f"Processing {num_prompts} prompts from dataset '{dataset_name}'")

    for i in tqdm(range(num_prompts), desc=f"Processing prompts ({generation_method})"):
        try:
            prompt = dataset[i] 
            inputs = tokenizer([prompt], return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(model.device)

            start_time = time.time()
            
            if generation_method == 'hf':
                output_ids = run_hf_generation(model, tokenizer, input_ids, max_new_tokens=128)
                al_mean = 0.0
            elif generation_method == 'eagle' or generation_method == 'mtp' or generation_method == 'bmeagle':
                output_ids, al = run_eagle_generation(model, input_ids, max_new_tokens=128)
                al_tensor = torch.as_tensor(al).float()
                al_mean = al_tensor.mean().item()
            elif generation_method == 'deepspeed':
                output_ids = run_deepspeed_generation(model, tokenizer, input_ids, max_new_tokens=128)
                al_mean = 0.0  # DeepSpeed standard generation doesn't have an AL metric
            elif generation_method == 'bm':
                output_ids = run_bm_generation(model, input_ids, max_new_tokens=128)
                al_mean = 0.0

            end_time = time.time()

            # Decode the generated output
            output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            total_time += (end_time - start_time)
            total_al_mean += al_mean
            
            if i < 3:
                print(output)

        except Exception as e:
            print(f"Error processing prompt {i}: {str(e)}")
            raise e

    if num_prompts > 0:
        average_time = total_time / num_prompts
        average_al = total_al_mean / num_prompts

        print(f"\n=== Results for {generation_method.upper()} method on {dataset_name} dataset ===")
        print(f"Average time across {num_prompts} prompts: {average_time:.4f} seconds")
        if generation_method in ['eagle', 'mtp']:
            print(f"Average AL (Acceptance Length) across {num_prompts} prompts: {average_al:.4f}")
        
        # Get memory usage - handle different cases
        try:
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                print(f"Peak GPU memory usage: {peak_memory / 1024**3:.2f} GB")
            else:
                print("No GPU available for memory measurement")
        except Exception as e:
            print(f"Could not measure memory usage: {str(e)}")
    else:
        print("No prompts were processed successfully")

if __name__ == "__main__":
    main()
