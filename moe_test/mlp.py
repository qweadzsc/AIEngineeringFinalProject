import torch
import torch.nn as nn
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


def sparsemixer(scores, top_k, jitter_eps):
    assert top_k == 2
    
    ################ first expert ################
    
    with torch.no_grad():
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts = max_ind

    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
    multiplier = multiplier_o

    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts_top2 = max_ind
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
    
    multiplier_top2 = multiplier_top2_o
    
    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)
    
    return (
        multiplier, 
        selected_experts,
    )


def mysparsemixer(scores, jitter_eps):
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts = max_ind
        
    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    # masked out first expert 
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    selected_experts_top2 = max_ind
    
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
    
    # Create output tensor with shape (batch_size, num_experts)
    multiplier = torch.zeros_like(scores)
    
    # Set the weights for the selected experts
    multiplier.scatter_(dim=-1, index=selected_experts, src=multiplier_o)
    multiplier.scatter_(dim=-1, index=selected_experts_top2, src=multiplier_top2_o)
    
    # Compute mask_c: sort experts by sum of scores along dim=0, return indices
    score_sums = torch.sum(scores, dim=0)
    _, mask_c = torch.sort(score_sums, descending=True)
    mask_c = mask_c.to(torch.int64)
    
    # Compute mask_v: get topk indices from score[:, mask_c[8:]]
    mask_v = torch.topk(scores[:, mask_c[8:]], k=4, dim=0).indices
    mask_v = mask_v.to(torch.int64)
    
    return multiplier, mask_c, mask_v


def opt_mixer(scores, jitter_eps):
    scores_abs = scores.abs() 

    with torch.no_grad():
        mask_logits_threshold, selected_experts = scores.max(dim=-1, keepdim=True)
        factor = scores_abs.clamp(min=mask_logits_threshold)
        mask_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)
    masked_gates = scores.masked_fill(mask_threshold, float('-inf'))
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
    masked_scores = scores.clone()
    masked_scores.scatter_(-1, selected_experts, float('-inf'))

    with torch.no_grad():
        mask_logits_threshold2, selected_experts_top2 = masked_scores.max(dim=-1, keepdim=True)
        factor2 = scores_abs.clamp(min=mask_logits_threshold2)
        mask_threshold2 = ((mask_logits_threshold2 - masked_scores) / factor2) > (2 * jitter_eps)
    masked_gates_top2 = masked_scores.masked_fill(mask_threshold2, float('-inf'))
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
    multiplier = torch.zeros_like(scores)
    multiplier.scatter_(dim=-1, index=selected_experts, src=multiplier_o)
    multiplier.scatter_(dim=-1, index=selected_experts_top2, src=multiplier_top2_o)

    score_sums = torch.sum(scores, dim=0)
    _, mask_c = torch.sort(score_sums, descending=True)
    mask_c = mask_c.to(torch.int64)
    relevant_indices = mask_c[8:]
    mask_v = torch.topk(scores[:, relevant_indices], k=4, dim=0).indices
    mask_v = mask_v.to(torch.int64)
    
    return multiplier, mask_c, mask_v


class SPMLP(nn.Module):
    def __init__(self, origin_mlp, t_d=None):
        super().__init__()

        with torch.no_grad():
            self.hidden_size = origin_mlp.hidden_dim
            self.num_experts = origin_mlp.num_experts
            self.top_k = origin_mlp.top_k
            self.norm_topk_prob = getattr(origin_mlp, 'norm_topk_prob', False)
            
            first_expert = origin_mlp.experts[0]
            self.intermediate_size = first_expert.ffn_dim
            self.total_intermediate_size = self.intermediate_size * self.num_experts
            self.act_fn = first_expert.act_fn  # silu
            dtype = first_expert.w1.weight.dtype
            device = first_expert.w1.weight.device
            
            self.combined_w1_weight = torch.empty(self.total_intermediate_size, self.hidden_size,
                                                  dtype=dtype, device=device)
            self.combined_w3_weight = torch.empty(self.total_intermediate_size, self.hidden_size,
                                                  dtype=dtype, device=device)
            self.combined_w2_weight = torch.empty(self.hidden_size, self.total_intermediate_size,
                                                  dtype=dtype, device=device)

            start_idx = 0
            for i, expert in enumerate(origin_mlp.experts):
                end_idx = start_idx + self.intermediate_size

                self.combined_w1_weight[start_idx:end_idx, :] = expert.w1.weight.data
                self.combined_w3_weight[start_idx:end_idx, :] = expert.w3.weight.data
                self.combined_w2_weight[:, start_idx:end_idx] = expert.w2.weight.data

                start_idx = end_idx

                del expert.w1.weight
                del expert.w2.weight
                del expert.w3.weight

            del origin_mlp.experts
            gc.collect()
            torch.cuda.empty_cache()

        self.gate = origin_mlp.gate
        self.t_d = self.num_experts // 2 if t_d is None else t_d
    
    def original_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = sparsemixer(
            router_logits, 
            top_k=2, 
            jitter_eps=0.01,
        )
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            start_idx = expert_idx * self.intermediate_size
            end_idx = start_idx + self.intermediate_size
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_up = torch.matmul(current_state, self.combined_w3_weight[start_idx:end_idx, :].t())
            current_gate = torch.matmul(current_state, self.combined_w1_weight[start_idx:end_idx, :].t())
            current_activation = self.act_fn(current_gate)
            current_result = current_activation * current_up
            current_result = torch.matmul(current_result, self.combined_w2_weight[:, start_idx:end_idx].t())
            current_result = current_result * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_result.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        bs = x.size(0)

        if bs not in [32, 64, 128]:
            return self.original_forward(hidden_states)
        
        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)
        # sparsemixer_start = torch.cuda.Event(enable_timing=True)
        # sparsemixer_end = torch.cuda.Event(enable_timing=True)

        # start_event.record()

        router_logits = self.gate(x)
        with torch.no_grad():
            # sparsemixer_start.record()
            router_weights, mask_c, mask_r = opt_mixer(router_logits, 0.01)
            # sparsemixer_end.record()

            ir = torch.zeros((2, bs, self.intermediate_size*self.t_d), device=x.device, dtype=x.dtype)
            mask_v = torch.zeros((2, 4, self.t_d*self.intermediate_size), device=x.device, dtype=x.dtype)
            result = torch.zeros((self.t_d, bs, self.hidden_size), device=x.device, dtype=x.dtype)

            mlp_kernel.ops.sddmm(x, self.combined_w3_weight, self.combined_w1_weight, ir, mask_r, mask_c, mask_v,
                                 bs, hidden_dim, self.total_intermediate_size, self.intermediate_size, self.t_d, 4)
            
            mlp_kernel.ops.spmm(ir[0], self.combined_w2_weight, result, mask_r, mask_c, mask_v, router_weights,
                                bs, hidden_dim, self.total_intermediate_size, self.intermediate_size, self.t_d, 4)
            x = result.sum(0)

        # end_event.record()

        # torch.cuda.synchronize()
        # total_time = start_event.elapsed_time(end_event)
        # sparsemixer_time = sparsemixer_start.elapsed_time(sparsemixer_end)
        # other_time = total_time - sparsemixer_time
        # sparsemixer_ratio = (sparsemixer_time / total_time) * 100 if total_time > 0 else 0
        # other_ratio = (other_time / total_time) * 100 if total_time > 0 else 0

        # print(f"Total time: {total_time:.4f} ms")
        # print(f"mysparsemixer time: {sparsemixer_time:.4f} ms ({sparsemixer_ratio:.2f}%)")
        # print(f"Other operations time: {other_time:.4f} ms ({other_ratio:.2f}%)")

        hidden_states = x.reshape((batch_size, sequence_length, hidden_dim))
        return hidden_states, router_logits
