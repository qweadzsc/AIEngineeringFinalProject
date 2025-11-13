import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import *
from torch.utils.cpp_extension import load
from mlp import MLPCUDA


def input_format(data):
    if len(data['input']) == 0:
        input_text = instruction_text_format.format(data['instruction'])
    else:
        input_text = instruction_input_text_format.format(data['instruction'], data['input'])
    data['input_text'] = input_text
    return data


def profile_batched_gen(model, tokenizer, ds, batch_size, device, n_run=100):
    total_time = 0
    for i in tqdm(range(n_run)):
        input_text = [ds[j]['input_text'] for j in range(batch_size*i, batch_size*(i+1))]
        input_ids = tokenizer(input_text, return_tensors='pt', padding='max_length',
                              max_length=64, truncation=True).to(device)
        start = time.time()
        output = model.generate(input_ids.input_ids, attention_mask=input_ids.attention_mask,
                                max_new_tokens=64, do_sample=False)
        eps = time.time() - start
        total_time += eps
        # print(tokenizer.decode(output[0]))
        # exit(-1)
        # print(eps)
    print("throughout:", total_time/n_run)


if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    model_path = "/share/public/zhouyongkang/models/"
    model_name = "TurboSparse-7B-mdf"

    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(model_path+model_name,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path+model_name)
    model = model.to(device)

    cutlass_path = "/share/public/zhouyongkang/projects/sc/deps/cutlass"
    dsmm_path = "/share/public/zhouyongkang/projects/sc/src/dsmm"
    build_path = "/share/public/zhouyongkang/projects/sc/src/build"
    cuda_module = load(name="mlp_kernel",
                       sources=["api.cu", "impl.cu", "dsmm/dense_sddmm.cu", "dsmm/dense_spmm.cu"],
                       extra_include_paths=[f'{cutlass_path}/include', dsmm_path],
                       build_directory=build_path,
                       verbose=True)
    for layer in model.model.layers:
        layer.mlp = MLPCUDA(layer.mlp, cuda_module)

    ds = load_dataset("/share/public/zhouyongkang/projects/sc/data/benchmark/alpaca", split='train')
    pds = ds.map(input_format, remove_columns=ds.column_names)
    profile_batched_gen(model, tokenizer, pds, 1024, device, 50)
    # for batch_size in (1, 2, 4, 8, 16, 32, 64):
    #     profile_batched_gen(model, tokenizer, pds, batch_size, device)
