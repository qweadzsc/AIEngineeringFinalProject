import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import mlp_kernel


import torch

def print_diff(tensor1, tensor2, eps=3e-3):
    """
    Compares two tensors and prints the percentage of elements with an absolute difference
    greater than a calculated threshold.

    Args:
        tensor1 (torch.Tensor): The first input tensor.
        tensor2 (torch.Tensor): The second input tensor.
        eps (float): A small value used to calculate the threshold.

    Raises:
        AssertionError: If the shapes of the two input tensors do not match.
    """
    if tensor1.shape != tensor2.shape:
        raise AssertionError(f"Tensor shapes do not match: {tensor1.shape} vs {tensor2.shape}")

    max_abs_value = torch.maximum(tensor1.abs().max(), tensor2.abs().max())
    threshold = max_abs_value * eps

    abs_diff = (tensor1 - tensor2).abs()
    diff_exceeds_threshold = abs_diff > threshold
    num_exceeding = diff_exceeds_threshold.sum().item()
    total_elements = tensor1.numel()

    percentage = (num_exceeding / total_elements) * 100
    print(f"Percentage of elements with absolute difference greater than threshold ({threshold:.2e}): {num_exceeding}/{total_elements} = {percentage:.2f}%, max: {abs_diff.max().item()}")


class SPMLP(nn.Module):
    def __init__(self, origin_mlp, t_d=None):
        super().__init__()

        with torch.no_grad():
            self.hidden_size = origin_mlp.experts[0].hidden_size
            self.num_experts = origin_mlp.num_experts
            self.top_k = origin_mlp.top_k
            self.norm_topk_prob = getattr(origin_mlp, 'norm_topk_prob', False)
            
            first_expert = origin_mlp.experts[0]
            self.intermediate_size = first_expert.intermediate_size
            self.total_intermediate_size = self.intermediate_size * self.num_experts
            self.act_fn = first_expert.act_fn
            dtype = first_expert.gate_proj.weight.dtype
            device = first_expert.gate_proj.weight.device
            
            self.combined_w1_weight = torch.empty(self.total_intermediate_size, self.hidden_size,
                                                  dtype=dtype, device=device)
            self.combined_w3_weight = torch.empty(self.total_intermediate_size, self.hidden_size,
                                                  dtype=dtype, device=device)
            self.combined_w2_weight = torch.empty(self.hidden_size, self.total_intermediate_size,
                                                  dtype=dtype, device=device)

            start_idx = 0
            for i, expert in enumerate(origin_mlp.experts):
                end_idx = start_idx + self.intermediate_size

                self.combined_w1_weight[start_idx:end_idx, :] = expert.gate_proj.weight.data
                self.combined_w3_weight[start_idx:end_idx, :] = expert.up_proj.weight.data
                self.combined_w2_weight[:, start_idx:end_idx] = expert.down_proj.weight.data

                start_idx = end_idx

                del expert.gate_proj.weight
                del expert.up_proj.weight
                del expert.down_proj.weight

            del origin_mlp.experts
            gc.collect()
            torch.cuda.empty_cache()

        self.gate = origin_mlp.gate
        self.t_d = self.num_experts // 2 if t_d is None else t_d
        
    def opt_mixer(self, router_logits):
        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_values, dim=1)  # size: (bs, k)
        router_weights = torch.zeros_like(router_logits)  # size: (bs, ne)
        router_weights.scatter_(1, top_k_indices, top_k_weights)
        
        col_sums = router_weights.sum(dim=0)  # size: (ne,)
        _, sorted_indices = torch.sort(col_sums, descending=True)

        half_size = self.t_d
        mask_c = sorted_indices[-half_size:]  # size: (t_d,)

        selected_weights = router_weights[:, mask_c]  # size: (bs, t_d)
        _, top_4_indices = torch.topk(selected_weights, 4, dim=0)  # size: (4, t_d)

        return router_weights, sorted_indices, top_4_indices
    
    def original_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            start_idx = expert_idx * self.intermediate_size
            end_idx = start_idx + self.intermediate_size
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert using combined weights
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            
            # Compute the expert operations using combined weights
            current_up = torch.matmul(current_state, self.combined_w3_weight[start_idx:end_idx, :].t())
            current_gate = torch.matmul(current_state, self.combined_w1_weight[start_idx:end_idx, :].t())
            current_activation = self.act_fn(current_gate)
            current_result = current_activation * current_up
            current_result = torch.matmul(current_result, self.combined_w2_weight[:, start_idx:end_idx].t())
            
            # Multiply by routing weights
            current_result = current_result * routing_weights[top_x, idx, None]

            # Add to final hidden states
            final_hidden_states.index_add_(0, top_x, current_result.to(hidden_states.dtype))
            
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        bs = x.size(0)

        if bs not in [32, 64, 128]:
            return self.original_forward(hidden_states)

        router_logits = self.gate(x)
        with torch.no_grad():
            router_weights, mask_c, mask_r = self.opt_mixer(router_logits)

            ir = torch.zeros((2, bs, self.intermediate_size*self.t_d), device=x.device, dtype=x.dtype)
            mask_v = torch.zeros((2, 4, self.t_d*self.intermediate_size), device=x.device, dtype=x.dtype)
            result = torch.zeros((self.t_d, bs, self.hidden_size), device=x.device, dtype=x.dtype)

            mlp_kernel.ops.sddmm(x, self.combined_w3_weight, self.combined_w1_weight, ir, mask_r, mask_c, mask_v,
                                 bs, hidden_dim, self.total_intermediate_size, self.intermediate_size, self.t_d, 4)
            
            mlp_kernel.ops.spmm(ir[0], self.combined_w2_weight, result, mask_r, mask_c, mask_v, router_weights,
                                bs, hidden_dim, self.total_intermediate_size, self.intermediate_size, self.t_d, 4)
            x = result.sum(0)

        hidden_states = x.reshape((batch_size, sequence_length, hidden_dim))
        return hidden_states, router_logits
    
    def bm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        ffn_dim_per_expert = self.intermediate_size
        num_experts = self.num_experts
        BSL = batch_size * sequence_length
        
        expanded_hidden_states = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1)
        
        w1_weights = self.combined_w1_weight.view(num_experts, ffn_dim_per_expert, hidden_dim).transpose(-2, -1)
        w3_weights = self.combined_w3_weight.view(num_experts, ffn_dim_per_expert, hidden_dim).transpose(-2, -1)
        w2_weights = self.combined_w2_weight.transpose(-2, -1).view(num_experts, ffn_dim_per_expert, hidden_dim)

        expert_inputs = torch.zeros(
            (num_experts, BSL, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        
        expanded_selected_experts = selected_experts.unsqueeze(-1).expand(-1, -1, hidden_dim)
        src_for_scatter = expanded_hidden_states.transpose(0, 1)
        index_for_scatter = expanded_selected_experts.transpose(0, 1)
        expert_inputs.scatter_add_(0, index_for_scatter, src_for_scatter)

        gate_outputs = torch.bmm(expert_inputs, w1_weights)
        w3_outputs = torch.bmm(expert_inputs, w3_weights)
        activated_gate_outputs = self.act_fn(gate_outputs)
        up_outputs = activated_gate_outputs * w3_outputs
        expert_outputs = torch.bmm(up_outputs, w2_weights)

        token_indices = torch.arange(batch_size * sequence_length, device=hidden_states.device).unsqueeze(1).expand(-1, self.top_k)

        final_expert_outputs_for_tokens = expert_outputs[selected_experts, token_indices]

        weighted_outputs = final_expert_outputs_for_tokens * routing_weights.unsqueeze(-1)

        final_hidden_states = weighted_outputs.sum(dim=1)

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits
