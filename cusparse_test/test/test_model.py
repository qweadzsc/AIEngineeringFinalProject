import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

model_path_turbo = "/share/zhouyongkang/model/TurboSparse-Mistral-Instruct"
model_path_relullama = "/share/zhouyongkang/model/ReluLLaMA-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path_turbo)
model = AutoModelForCausalLM.from_pretrained(model_path_turbo, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
dataset = load_dataset("/share/zhouyongkang/Benchmark/Alpaca")
data = dataset["train"]

in_text_batch = []
bs = 64
tokenizer.pad_token = tokenizer.eos_token

# for i in tqdm(range(10)):
# for i in range(64):
#     in_text = data[i]["instruction"] + "\n" + data[i]["input"]
#     in_text_batch.append(in_text)
#     if len(in_text_batch) != bs:
#         continue
#     in_ids = tokenizer(in_text_batch, return_tensors="pt", padding=True)
#     output = model.generate(in_ids["input_ids"].to('cuda'), max_new_tokens=100, do_sample=False, attention_mask=in_ids["attention_mask"].to('cuda'))
    # print(tokenizer.decode(output))

for i in range(10):
    in_text = data[i]["instruction"] + "\n" + data[i]["input"]
    in_ids = tokenizer(in_text, return_tensors="pt")
    output = model.generate(in_ids["input_ids"].to('cuda'), max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(output[0]))
