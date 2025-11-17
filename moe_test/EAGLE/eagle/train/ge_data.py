import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=20000)
parser.add_argument('--index', type=str, default='qwen3-8b-base-draft')
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='/share/public/zhouyongkang/projects/sc/moe_test/EAGLE/eagle/train/data')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig,Qwen3ForCausalLM
from datasets import load_dataset,Dataset
import json
from fastchat.model.model_adapter import get_conversation_template
bigname="/share/others/public_models/Qwen3-14B/"
model = AutoModelForCausalLM.from_pretrained("/share/others/public_models/Qwen3-14B/",torch_dtype=torch.float32,device_map="auto")
model.eval()

def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
        json_data=None,
):
    ds = load_dataset('json', data_files=json_data, split="train", streaming=True)

    # 流式打乱（需datasets>=2.0版本）
    ds_shuffled = ds.shuffle(seed=42, buffer_size=10000)

    # 关键：先将流式数据转换为列表（限制最大加载量为20000+，确保有足够数据）
    # 取20000+args.start条数据，避免后续切片越界
    max_samples = 100000 + args.start
    data_list = list(ds_shuffled)  # 转换为列表，支持len()

    # 检查长度并切片（替代select操作）
    if len(data_list) > 100000:
        # 从args.start开始，取到20000结束（左闭右开）
        start_idx = args.start
        end_idx = min(100000, len(data_list))  # 防止超出实际长度
        data_list = data_list[start_idx:end_idx]

    # 转换回Dataset对象，保留数据集方法
    ds1 = Dataset.from_list(data_list)
    original_columns1 = ds1.column_names
    # original_columns2 = ds2.column_names
    num_proc = 1
    # breakpoint()

    def preprocess_function(examples):
        # breakpoint()
        new_examples = {
            "input_ids": [],
        }
        # breakpoint()
        for i in range(len(examples['text'])):
            turns = examples['text'][i]
            tokenized = tokenizer(
                turns,
                return_tensors="pt",
                # max_length=,
                truncation=True,
            )
            input_ids = tokenized.input_ids[0]
            attn_mask = tokenized.attention_mask[0]
            # breakpoint()
            # logits = model(input_ids.unsqueeze(0)).logits
            # prob = torch.softmax(logits,dim=-1)
            # out_tokens = torch.argmax(prob,dim=-1)[0]
            # # breakpoint()
            # if input_ids.shape[0] >= 2048:
            #     new_input_ids = input_ids[:2048]
            #     # labels = out_tokens
            # else:
            #     new_input_ids = F.pad(input_ids, pad=(0, 2048-input_ids.shape[0]), mode='constant', value=151643)
            #     labels = F.pad(out_tokens, pad=(0, 2048-out_tokens.shape[0]), mode='constant', value=-100)
                # breakpoint()

            # breakpoint()
            new_examples["input_ids"].append(input_ids[None,:])
            # new_examples["labels"].append(labels[None,:])


        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )
    ds1.set_format(type="torch")
    return ds1

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

@torch.no_grad()


def ge(data):
    input_ids=data["input_ids"][0].cuda()
    # labels=data["labels"][0]
    if input_ids.shape[0] >= 2048:
        new_input_ids = input_ids[:2048]
    else:
        new_input_ids = F.pad(input_ids, pad=(0, 2048-input_ids.shape[0]), mode='constant', value=151643)
    output = model(new_input_ids.unsqueeze(0))
    logits = output.logits
    # breakpoint()
    hidden_states = output.hidden_states[0].cpu()
    prob = torch.softmax(logits,dim=-1)
    out_tokens = torch.argmax(prob,dim=-1)[0]
    loss_mask = torch.ones(2048)
    # breakpoint()
    if input_ids.shape[0] >= 2048:
        # new_input_ids = input_ids[:2048]
        labels = out_tokens[:2048]
        hidden_states = hidden_states[:2048,:]
    else:
        loss_mask[hidden_states.shape[0]:] = 0
        # new_input_ids = F.pad(input_ids, pad=(0, 2048-input_ids.shape[0]), mode='constant', value=151643)
        labels = F.pad(out_tokens, pad=(0, 2048-out_tokens.shape[0]), mode='constant', value=-100)
        padding = (0, 0, 0, 2048 - hidden_states.shape[0])  # 即 (0, 0, 0, 2000)
        hidden_states = F.pad(hidden_states, padding, mode='constant', value=0)
        # breakpoint()    
    # attention_mask = data["new_attn_mask"][0]
    # breakpoint()
    # labels = data["labels"][0][1:]
    # labels = F.pad(labels, pad=(0, 1), mode='constant', value=-100)
    # breakpoint()
    td={"input_ids":new_input_ids.cpu(),"labels":labels.cpu(),"hidden_states":hidden_states.cpu(),"loss_mask":loss_mask.cpu()}
    # breakpoint()
    return td

bigtokenizer = AutoTokenizer.from_pretrained(bigname,use_fast=False)
# breakpoint()
filenames = ["/home/panjiayi/panjiayi/SpecEE落地/train_data/sample_data/sampled_redpajama-cc-2023-06-refine-result.jsonl"]
# filenames = ["/home/panjiayi/panjiayi/SpecEE落地/train_data/sample_data/sampled_the-pile-hackernews-refine-result.jsonl"]
for filename in filenames:
    # print(filename)
    if filename.endswith('.jsonl'):
        ds = build_dataset_rank(bigtokenizer,json_data=filename)
        # breakpoint()
        print(ds)

        outdir = f'{args.outdir}/{args.index}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for id, data in enumerate(tqdm(ds, desc="Processing", total=len(ds))): # 添加total=len(ds)让进度条更准确
            outdata = ge(data)
            writedata(outdir, outdata)
