import argparse
import os
import time

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data import CustomTextDataset, resolve_dataset_path
from EAGLE.eagle.model.ea_model import EaModel


MODEL_PRESETS = {
    "qwen3_30b": {
        "base_model_path": "/share/public/public_models/Qwen3-30B-A3B",
        "eagle_model_path": "/share/zhouyongkang/models/qwen3_30b_moe_eagle3",
    },
    "gpt_oss_20b": {
        "base_model_path": "/share/public/public_models/gpt-oss-20b",
        "eagle_model_path": "/share/zhouyongkang/models/EAGLE3-gpt-oss-20b",
    },
}


def resolve_dtype(dtype_name: str, model_type: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name != "auto":
        raise ValueError(f"Unsupported dtype specifier: {dtype_name}")
    return torch.bfloat16 if model_type == "gpt_oss" else torch.float16


def get_model_device(model) -> torch.device:
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.layers[0].self_attn.q_proj.weight.device
    return next(model.parameters()).device


def build_model_inputs(tokenizer, prompt: str, model_type: str, input_format: str):
    use_chat = input_format == "chat" or (input_format == "auto" and model_type == "gpt_oss")
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        if hasattr(rendered, "keys"):
            model_inputs = {key: value for key, value in rendered.items()}
        else:
            model_inputs = {"input_ids": rendered, "attention_mask": torch.ones_like(rendered)}
    else:
        rendered = tokenizer([prompt], return_tensors="pt", padding=True)
        model_inputs = {key: value for key, value in rendered.items()}

    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
    return model_inputs


def run_hf_generation(model, tokenizer, model_inputs, max_new_tokens, temperature=0.0, top_p=None, top_k=None):
    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
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
    output_ids, al = model.eagenerate(input_ids, max_new_tokens=max_new_tokens)
    return output_ids, al


def run_bm_generation(model, input_ids, max_new_tokens):
    output_ids = model.naivegenerate(input_ids, max_new_tokens=max_new_tokens)
    return output_ids


def run_deepspeed_generation(model, tokenizer, model_inputs, max_new_tokens, temperature=0.0, top_p=None, top_k=None):
    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
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


def load_deepspeed_model(base_model_path, dtype=torch.float16):
    import deepspeed

    print(f"Loading model for DeepSpeed inference from {base_model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    ds_config = {
        "dtype": dtype,
        "replace_method": "auto",
        "replace_with_kernel_inject": True,
        "enable_cuda_graph": False,
    }

    if "moe" in base_model_path.lower() or hasattr(model.config, "num_experts") or hasattr(model.config, "num_local_experts"):
        ds_config["replace_with_kernel_inject"] = False
        print("Warning: MoE models may not support kernel injection. Disabling it.")

    ds_engine = deepspeed.init_inference(model=model, config=ds_config)
    model = ds_engine.module
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def collect_spmlp_stats(model):
    base_model = getattr(model, "base_model", None)
    model_body = getattr(base_model, "model", None)
    layers = getattr(model_body, "layers", None)
    if layers is None:
        return None

    totals = {}
    counted_layers = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        stats_fn = getattr(mlp, "get_path_stats", None)
        if not callable(stats_fn):
            continue
        counted_layers += 1
        for key, value in stats_fn().items():
            totals[key] = totals.get(key, 0) + int(value)

    if counted_layers == 0:
        return None

    totals["spmlp_layers"] = counted_layers
    return totals


def main():
    parser = argparse.ArgumentParser(description="Run LLM generation with different methods.")
    parser.add_argument(
        "--dataset",
        type=int,
        default=7,
        help="Index of the dataset in ['alpaca', 'commonsense_qa', 'gsm8k', 'hellaswag', 'piqa', 'siqa', 'sst2', 'sum'] (default: 7)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hf",
        choices=["hf", "eagle", "mtp", "deepspeed", "bm", "bmeagle"],
        help="Generation method: hf, eagle, mtp, bm, bmeagle, or deepspeed (default: hf)",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="qwen3_30b",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Predefined model family used to fill default model paths. (default: qwen3_30b)",
    )
    parser.add_argument("--base-model-path", type=str, default=None, help="Override base model path.")
    parser.add_argument("--eagle-model-path", type=str, default=None, help="Override EAGLE draft model path.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="Model dtype. auto uses bfloat16 for GPT-OSS and float16 otherwise. (default: auto)",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "plain", "chat"],
        help="Prompt formatting style. auto uses chat template for GPT-OSS and plain text otherwise. (default: auto)",
    )
    parser.add_argument(
        "--num-prompts",
        dest="num_prompts",
        type=int,
        default=30,
        help="Number of prompts to process. (default: 30)",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=128,
        help="Max generated tokens for each prompt. (default: 128)",
    )
    parser.add_argument(
        "--spmlp-t-d",
        type=int,
        default=32,
        help="Configured t_d used by SPMLP. When adaptive td is disabled this stays fixed. (default: 32)",
    )
    parser.add_argument(
        "--spmlp-adaptive-td",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether SPMLP should overwrite t_d at runtime using count_nonzero(expert_hit_count > maxnnz). (default: True)",
    )
    parser.add_argument(
        "--spmlp-bm-fallback-td",
        type=int,
        default=32,
        help="Runtime t_d threshold used by the main-path fallback branch. Set a negative value to disable threshold fallback. (default: 32)",
    )
    parser.add_argument(
        "--spmlp-maxnnz",
        type=int,
        default=4,
        help="maxnnz used by the SPMLP sparse row selection path. (default: 4)",
    )
    parser.add_argument(
        "--ea-total-token",
        type=int,
        default=64,
        help="Total drafted tokens budget passed to EaModel.from_pretrained(...). (default: 64)",
    )
    parser.add_argument(
        "--spmlp-unsupported-fallback-mode",
        type=str,
        default="bm",
        choices=["original", "bm"],
        help="Fallback used when batch_size * sequence_length is not supported by the CUDA main path. (default: bm)",
    )
    parser.add_argument(
        "--spmlp-runtime-fallback-mode",
        type=str,
        default="bm",
        choices=["none", "original", "bm"],
        help="Fallback used when runtime t_d exceeds the configured threshold. (default: bm)",
    )
    args = parser.parse_args()

    datasets_names = ["alpaca", "commonsense_qa", "gsm8k", "hellaswag", "piqa", "siqa", "sst2", "sum"]
    if args.dataset < 0 or args.dataset >= len(datasets_names):
        raise ValueError(f"Dataset index {args.dataset} is out of range for list of {len(datasets_names)} datasets.")

    dataset_name = datasets_names[args.dataset]
    dataset_path = resolve_dataset_path(f"benchmark/{dataset_name}")
    print(f"Loading dataset: {dataset_name} from {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path {dataset_path} does not exist. Attempting to load anyway...")
    dataset = CustomTextDataset(dataset_path)

    preset = MODEL_PRESETS[args.model_family]
    base_model_path = args.base_model_path or preset["base_model_path"]
    eagle_model_path = args.eagle_model_path or preset["eagle_model_path"]

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_type = config.model_type
    dtype = resolve_dtype(args.dtype, model_type)
    generation_method = args.method.lower()

    print(f"Selected generation method: {generation_method}")
    print(f"Base model path: {base_model_path}")
    print(f"EAGLE model path: {eagle_model_path}")
    print(f"Model type: {model_type}")
    print(f"Resolved dtype: {dtype}")
    print(f"Input format: {args.input_format}")

    spmlp_bm_fallback_t_d = None if args.spmlp_bm_fallback_td < 0 else args.spmlp_bm_fallback_td
    print(
        "SPMLP config: "
        f"t_d={args.spmlp_t_d}, adaptive_t_d={args.spmlp_adaptive_td}, "
        f"bm_fallback_t_d={spmlp_bm_fallback_t_d}, "
        f"maxnnz={args.spmlp_maxnnz}, "
        f"ea_total_token={args.ea_total_token}, "
        f"unsupported_fallback={args.spmlp_unsupported_fallback_mode}, "
        f"runtime_fallback={args.spmlp_runtime_fallback_mode}"
    )

    model = None
    tokenizer = None

    if generation_method == "hf":
        print(f"Loading standard Hugging Face model from {base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif generation_method == "eagle":
        print(f"Loading EAGLE model with base {base_model_path} and EA model {eagle_model_path}...")
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=eagle_model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=args.ea_total_token,
        )
        model.eval()
        model.device = get_model_device(model)
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif generation_method in {"mtp", "bm", "bmeagle"}:
        if model_type == "gpt_oss" and generation_method == "mtp":
            raise NotImplementedError("GPT-OSS currently only supports hf/eagle/bm/bmeagle in main.py.")

        from mlp import SPMLP

        print(f"Loading EAGLE model with base {base_model_path} and EA model {eagle_model_path} for SPMLP...")
        model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=eagle_model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            total_token=args.ea_total_token,
        )
        model.eval()
        model.device = get_model_device(model)
        forward_mode = "bm" if generation_method in {"bm", "bmeagle"} else "main"
        for layer in tqdm(model.base_model.model.layers, desc=f"Applying SPMLP ({forward_mode}) to layers"):
            layer.mlp = SPMLP(
                layer.mlp,
                forward_mode=forward_mode,
                t_d=args.spmlp_t_d,
                adaptive_t_d=args.spmlp_adaptive_td,
                bm_fallback_t_d=spmlp_bm_fallback_t_d,
                unsupported_batch_fallback_mode=args.spmlp_unsupported_fallback_mode,
                runtime_t_d_fallback_mode=args.spmlp_runtime_fallback_mode,
            )
            layer.mlp.maxnnz = args.spmlp_maxnnz
        tokenizer = model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif generation_method == "deepspeed":
        model, tokenizer = load_deepspeed_model(base_model_path, dtype=dtype)

    else:
        raise NotImplementedError(f"Unknown method {generation_method}")

    input_device = get_model_device(model)
    total_time = 0.0
    total_al_mean = 0.0
    num_prompts = min(args.num_prompts, len(dataset))
    print(f"Processing {num_prompts} prompts from dataset '{dataset_name}'")

    for i in tqdm(range(num_prompts), desc=f"Processing prompts ({generation_method})"):
        prompt = dataset[i]
        model_inputs = build_model_inputs(tokenizer, prompt, model_type=model_type, input_format=args.input_format)
        model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}
        input_ids = model_inputs["input_ids"]

        start_time = time.time()
        if generation_method == "hf":
            output_ids = run_hf_generation(model, tokenizer, model_inputs, max_new_tokens=args.max_new_tokens)
            al_mean = 0.0
        elif generation_method in {"eagle", "mtp", "bmeagle"}:
            output_ids, al = run_eagle_generation(model, input_ids, max_new_tokens=args.max_new_tokens)
            al_tensor = torch.as_tensor(al).float()
            al_mean = al_tensor.mean().item()
        elif generation_method == "deepspeed":
            output_ids = run_deepspeed_generation(model, tokenizer, model_inputs, max_new_tokens=args.max_new_tokens)
            al_mean = 0.0
        elif generation_method == "bm":
            output_ids = run_bm_generation(model, input_ids, max_new_tokens=args.max_new_tokens)
            al_mean = 0.0
        else:
            raise NotImplementedError(f"Unknown method {generation_method}")
        end_time = time.time()

        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        total_time += end_time - start_time
        total_al_mean += al_mean

        if i < 3:
            print(output)

    if num_prompts > 0:
        average_time = total_time / num_prompts
        average_al = total_al_mean / num_prompts

        print(f"\n=== Results for {generation_method.upper()} method on {dataset_name} dataset ===")
        print(f"Average time across {num_prompts} prompts: {average_time:.4f} seconds")
        if generation_method in ["eagle", "mtp", "bmeagle"]:
            print(f"Average AL (Acceptance Length) across {num_prompts} prompts: {average_al:.4f}")

        try:
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                print(f"Peak GPU memory usage: {peak_memory / 1024**3:.2f} GB")
            else:
                print("No GPU available for memory measurement")
        except Exception as e:
            print(f"Could not measure memory usage: {str(e)}")

        spmlp_stats = collect_spmlp_stats(model)
        if spmlp_stats is not None:
            print(
                "SPMLP path stats: "
                f"layers={spmlp_stats['spmlp_layers']} "
                f"kernel_path_calls={spmlp_stats['kernel_path_calls']} "
                f"unsupported_batch_fallback_calls={spmlp_stats['unsupported_batch_fallback_calls']} "
                f"runtime_t_d_fallback_calls={spmlp_stats['runtime_t_d_fallback_calls']} "
                f"original_forward_calls={spmlp_stats['original_forward_calls']} "
                f"bm_forward_calls={spmlp_stats['bm_forward_calls']}"
            )
    else:
        print("No prompts were processed successfully")


if __name__ == "__main__":
    main()
