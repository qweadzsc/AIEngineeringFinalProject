from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from tqdm import tqdm


model_path = "/share/public/zhouyongkang/models/"
model_name = "TurboSparse-7B-mdf"

model = AutoModelForCausalLM.from_pretrained(model_path+model_name, torch_dtype=torch.float16,
                                             trust_remote_code=True)
data = torch.load(
    "/share/public/zhouyongkang/projects/sc/data/gen_data/inter_result_mmlu.pt",
    weights_only=True
)
mask = (data > 0).to(torch.float32)
for i in tqdm(range(32)):
    layer = model.model.layers[i]
    mlp = layer.mlp
    up_w = mlp.up_proj.weight.data
    gate_w = mlp.gate_proj.weight.data
    down_w = mlp.down_proj.weight.data
    out_w = mlp.predictor.fc2.weight.data

    data_l = mask[i]
    act_rate = data_l.mean(dim=0)
    sorted_idx = act_rate.sort(descending=True).indices

    temp = up_w[sorted_idx].contiguous()
    mlp.up_proj.weight.data = temp
    temp = gate_w[sorted_idx].contiguous()
    mlp.gate_proj.weight.data = temp
    temp = down_w.T[sorted_idx].T.contiguous()
    mlp.down_proj.weight.data = temp
    temp = out_w[sorted_idx].contiguous()
    mlp.predictor.fc2.weight.data = temp

state_dict = model.state_dict()
with safe_open(model_path+model_name+"/mdf.safetensors", framework="pt") as f:
    metadata = f.metadata()
    save_file(state_dict, model_path+model_name+"/mdf-mmlu.safetensors", metadata)
