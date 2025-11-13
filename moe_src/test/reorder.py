import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
from collections import defaultdict
import shutil

def reorder_experts_by_activation_rate(model_path, activation_rates, save_path, num_experts=128):
    config_path = os.path.join(model_path, "config.json")
    import json
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(model_path, "model.safetensors.index.json"), 'r') as f:
        weight_map = json.load(f)["weight_map"]

    model_weights = {}

    safetensors_files = set(weight_map.values())
    for file_name in safetensors_files:
        file_path = os.path.join(model_path, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                model_weights[k] = f.get_tensor(k)
    
    # 3. 对每层进行expert重排
    for layer_idx in range(48):
        print(f"Processing layer {layer_idx}")
        
        # 获取当前层的激活率并排序（降序）
        layer_activation_rates = activation_rates[layer_idx]
        sorted_indices = torch.argsort(layer_activation_rates, descending=True)
        original_indices = torch.argsort(sorted_indices)
        
        # 收集当前层所有expert的权重键名
        expert_weight_keys = []
        gate_weight_keys = []
        
        # 找到当前层相关的权重键
        for key in model_weights.keys():
            if f"model.layers.{layer_idx}.mlp.experts." in key:
                expert_weight_keys.append(key)
            elif f"model.layers.{layer_idx}.mlp.gate." in key:
                gate_weight_keys.append(key)
        
        # 4. 重排expert权重
        # 按照权重类型分组处理
        expert_groups = defaultdict(dict)
        for key in expert_weight_keys:
            # 解析expert索引
            parts = key.split('.')
            expert_idx = None
            for i, part in enumerate(parts):
                if part == 'experts' and i + 1 < len(parts):
                    try:
                        expert_idx = int(parts[i + 1])
                        experts_pos = i
                        break
                    except ValueError:
                        continue
            
            if expert_idx is not None:
                weight_type_parts = parts.copy()
                weight_type_parts[experts_pos + 1] = '{}'  # 用占位符替换expert索引
                weight_type = '.'.join(weight_type_parts)
                expert_groups[weight_type][expert_idx] = model_weights[key]
        
        updates = {}  # 收集所有更新

        for weight_type, expert_tensors in expert_groups.items():
            if len(expert_tensors) == num_experts:
                # 将tensor按排序后的顺序重新排列
                sorted_tensors = [expert_tensors[sorted_indices[i].item()] for i in range(num_experts)]
                
                # 收集更新操作
                for i, tensor in enumerate(sorted_tensors):
                    # 构造新的键名
                    new_key = weight_type.replace('{}', str(i))
                    updates[new_key] = tensor

        # 执行更新
        for key, tensor in updates.items():
            model_weights[key] = tensor
        
        # 5. 重排gate权重
        for gate_key in gate_weight_keys:
            if 'weight' in gate_key:
                # gate权重的第一维对应expert数量，需要重排
                assert model_weights[gate_key].shape[0] == num_experts
                model_weights[gate_key] = model_weights[gate_key][sorted_indices]
            elif 'bias' in gate_key:
                # gate偏置也需要重排
                assert model_weights[gate_key].shape[0] == num_experts
                model_weights[gate_key] = model_weights[gate_key][sorted_indices]
    
    # 6. 保存模型
    # 创建新的weight_map
    new_weight_map = {}
    safetensors_files = set(weight_map.values())
    
    # 重新分配权重到对应的文件
    file_weights = defaultdict(dict)
    for key, original_file in weight_map.items():
        file_weights[original_file][key] = model_weights[key]
        new_weight_map[key] = original_file
    
    # 保存每个分片文件
    os.makedirs(save_path, exist_ok=True)
    for file_name, weights in file_weights.items():
        save_file(weights, os.path.join(save_path, file_name))
    
    # 保存index文件
    index_data = {
        "metadata": {"total_size": sum(os.path.getsize(os.path.join(save_path, f)) for f in file_weights.keys())},
        "weight_map": new_weight_map
    }
    with open(os.path.join(save_path, "model.safetensors.index.json"), 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # 7. 复制所有相关文件到目标目录
    files_to_copy = [
        # "config.json",
        # "merges.txt",
        # "tokenizer_config.json",
        # "tokenizer.json",
        # "special_tokens_map.json",
        # "generation_config.json",
        # "vocab.json",
        # "pytorch_model.bin.index.json"  # 如果存在
    ]
    
    for file_name in files_to_copy:
        src_path = os.path.join(model_path, file_name)
        dst_path = os.path.join(save_path, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {file_name}")
    
    print(f"Model with reordered experts saved to {save_path}")

activation_rates = torch.load('/share/public/zhouyongkang/projects/sc/data/expert_activation_rates.pt').to('cpu')
omd = "/share/public/public_models/Qwen3-30B-A3B/"
rmd = "/share/public/zhouyongkang/models/Qwen3-30B-A3B-Reordered"
reorder_experts_by_activation_rate(omd, activation_rates, rmd)

# kd = 110
# adata = []
# for i in range(48):
#     ldata = []
#     for j in range(kd):
#         data = torch.load(f"/share/public/zhouyongkang/projects/sc/data/mmlu_qwen3moe/{i}-{j+1}0.pt")
#         ldata.append(data)
#     lcdata = torch.cat(ldata, dim=0).to(torch.float32).mean(dim=0)
#     adata.append(lcdata)
# acdata = torch.stack(adata)
# acdata /= acdata.sum(dim=1, keepdim=True)
# torch.save(acdata, '/share/public/zhouyongkang/projects/sc/data/expert_activation_rates.pt')
# import matplotlib.pyplot as plt
# import numpy as np
# for i in range(48):
#     lcdata = acdata[i]
#     lcdata = lcdata.sort(descending=True).values
#     clc = torch.cumsum(lcdata, 0)
#     total_sum = clc[-1]
#     threshold = 0.7 * total_sum
#     threshold_index = torch.where(clc > threshold)[0]
#     if len(threshold_index) > 0:
#         threshold_position = threshold_index[0].item()
#     else:
#         threshold_position = len(clc) - 1
#     sorted_data_np = lcdata.cpu().numpy()
#     cumulative_sum_np = clc.cpu().numpy()
#     x_axis = np.arange(len(sorted_data_np))
#     plt.plot(x_axis, sorted_data_np, 'b-', linewidth=2, marker='o', markersize=3)
#     plt.axvline(x=threshold_position, color='r', linestyle='--', linewidth=2)
#     plt.text(threshold_position, plt.ylim()[1] * 0.9, f'Index: {threshold_position}', 
#              verticalalignment='top', horizontalalignment='center',
#              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     plt.tight_layout()
#     plt.savefig(f"/share/public/zhouyongkang/projects/sc/moe_test/fig/hot_cold/{i}.png")
#     plt.clf()
