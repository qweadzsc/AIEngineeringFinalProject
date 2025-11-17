import torch
import torch.nn as nn
import gc

class SPMLP(nn.Module):
    def __init__(self, origin_mlp):
        super().__init__()

        with torch.no_grad():
            self.hidden_size = origin_mlp.config.hidden_size
            self.num_experts = origin_mlp.num_experts
            self.top_k = origin_mlp.top_k
            self.norm_topk_prob = origin_mlp.norm_topk_prob
            
            first_expert = origin_mlp.experts[0]
            self.intermediate_size = first_expert.intermediate_size
            self.total_intermediate_size = self.intermediate_size * self.num_experts
            self.act_fn = first_expert.act_fn
            dtype = first_expert.gate_proj.weight.dtype
            
            self.combined_gate_weight = torch.empty(self.total_intermediate_size, self.hidden_size, dtype=dtype)
            self.combined_up_weight = torch.empty(self.total_intermediate_size, self.hidden_size, dtype=dtype)
            self.combined_down_weight = torch.empty(self.hidden_size, self.total_intermediate_size, dtype=dtype)
            
            start_idx = 0
            for i, expert in enumerate(origin_mlp.experts):
                end_idx = start_idx + self.intermediate_size
                self.combined_gate_weight[start_idx:end_idx] = expert.gate_proj.weight.data 
                self.combined_up_weight[start_idx:end_idx] = expert.up_proj.weight.data
                self.combined_down_weight[:, start_idx:end_idx] = expert.down_proj.weight.data
                
                start_idx = end_idx
                del expert.gate_proj.weight
                del expert.up_proj.weight
                del expert.down_proj.weight

            del origin_mlp.experts
            gc.collect()
            torch.cuda.empty_cache()

            self.gate = origin_mlp.gate
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()