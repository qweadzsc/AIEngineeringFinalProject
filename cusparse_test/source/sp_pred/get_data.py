from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from tqdm import tqdm


instruction_text_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

instruction_input_text_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""


def input_format(data):
    if len(data['input']) == 0:
        input_text = instruction_text_format.format(data['instruction'])
    else:
        input_text = instruction_input_text_format.format(data['instruction'], data['input'])
    data['input_text'] = input_text
    return data


if __name__ == "__main__":
    model_path = "/share/xujiaming/train_machine/dataset/"
    model_name = "TurboSparse-7B-mdf"

    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(model_path+model_name, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path+model_name)
    model = model.to(device)

    ds = load_dataset("/share/xujiaming/train_machine/yongkang/cusparse_test/data/benchmark/alpaca", split='train')
    pds = ds.map(input_format, remove_columns=ds.column_names)

    for i in tqdm(range(500)):
        input_text = pds[i]['input_text']
        input_ids = tokenizer(input_text, return_tensors='pt').to(device)
        output = model.generate(input_ids.input_ids, attention_mask=input_ids.attention_mask, max_new_tokens=100, do_sample=False)
