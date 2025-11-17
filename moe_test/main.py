import torch
import time
from datasets import load_dataset
from tqdm import tqdm
from EAGLE.eagle.model.ea_model import EaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlp import SPMLP

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


dataset = load_dataset('/share/public/zhouyongkang/projects/sc/data/benchmark/alpaca')
split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
dataset_split = dataset[split_name]

base_model_path = "/share/others/public_models/Qwen3-30B-A3B"
EAGLE_model_path = "/share/public/zhouyongkang/models/qwen3_30b_moe_eagle3"

# Define the generation method
generation_method = input("Enter generation method: ").strip().lower()

model = None
tokenizer = None

if generation_method == 'hf':
    print(f"Loading standard Hugging Face model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Ensure pad_token is defined, often needed for generation
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
        total_token=64
    )
    model.eval()
    model.device = model.base_model.device
    tokenizer = model.tokenizer # Use tokenizer from EaModel

elif generation_method == 'mtp':
    print(f"Loading EAGLE model with base {base_model_path} and EA model {EAGLE_model_path}...")
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=64
    )
    model.eval()
    model.device = model.base_model.device
    for layer in tqdm(model.base_model.model.layers):
        layer.mlp = SPMLP(layer.mlp)
    tokenizer = model.tokenizer

else:
    raise NotImplementedError(f"Unknown method {generation_method}")


total_time = 0
total_al_mean = 0
num_prompts = min(30, len(dataset_split))

for i in tqdm(range(num_prompts), desc="Processing prompts"):
    prompt = dataset_split[i]['instruction']
    
    # Tokenize input using the loaded tokenizer
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device) # Ensure input is on the correct device

    start_time = time.time()
    
    if generation_method == 'hf':
        output_ids = run_hf_generation(model, tokenizer, input_ids, max_new_tokens=128)
        al_mean = 0.0 # Standard HF generation does not have an acceptance list metric
    elif generation_method == 'eagle' or 'mtp':
        output_ids, al = run_eagle_generation(model, input_ids, max_new_tokens=128)
        al_tensor = torch.as_tensor(al).float()
        al_mean = al_tensor.mean().item()
    
    end_time = time.time()

    # Decode the generated output (excluding the prompt if necessary, though generation methods usually handle this)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    total_time += (end_time - start_time)
    total_al_mean += al_mean

average_time = total_time / num_prompts
average_al = total_al_mean / num_prompts

print(f"Average time across {num_prompts} prompts: {average_time:.4f} seconds")
print(f"Average AL across {num_prompts} prompts: {average_al:.4f}")