import gc

import mlp_kernel
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    print(
        f"Percentage of elements with absolute difference greater than threshold ({threshold:.2e}): "
        f"{num_exceeding}/{total_elements} = {percentage:.2f}%, max: {abs_diff.max().item()}"
    )


class SPMLP(nn.Module):
    def __init__(
        self,
        origin_mlp,
        t_d=None,
        forward_mode="main",
        adaptive_t_d=True,
        bm_fallback_t_d=None,
        unsupported_batch_fallback_mode="bm",
        runtime_t_d_fallback_mode="bm",
    ):
        super().__init__()

        with torch.no_grad():
            self.hidden_size = origin_mlp.experts[0].hidden_size
            self.num_experts = origin_mlp.num_experts
            self.top_k = origin_mlp.top_k
            self.norm_topk_prob = getattr(origin_mlp, "norm_topk_prob", False)

            first_expert = origin_mlp.experts[0]
            self.intermediate_size = first_expert.intermediate_size
            self.total_intermediate_size = self.intermediate_size * self.num_experts
            self.act_fn = first_expert.act_fn
            dtype = first_expert.gate_proj.weight.dtype
            device = first_expert.gate_proj.weight.device

            self.combined_w1_weight = torch.empty(
                self.total_intermediate_size,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
            self.combined_w3_weight = torch.empty(
                self.total_intermediate_size,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
            self.combined_w2_weight = torch.empty(
                self.hidden_size,
                self.total_intermediate_size,
                dtype=dtype,
                device=device,
            )

            start_idx = 0
            for expert in origin_mlp.experts:
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

        self.w1_weight_by_expert = self.combined_w1_weight.view(
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        self.w3_weight_by_expert = self.combined_w3_weight.view(
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        self.w2_weight_by_expert = self.combined_w2_weight.transpose(-2, -1).view(
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
        )

        if forward_mode not in {"main", "bm"}:
            raise ValueError(f"Unsupported forward mode: {forward_mode}")
        if unsupported_batch_fallback_mode not in {"original", "bm"}:
            raise ValueError(
                f"Unsupported unsupported_batch_fallback_mode: {unsupported_batch_fallback_mode}"
            )
        if runtime_t_d_fallback_mode not in {"none", "original", "bm"}:
            raise ValueError(f"Unsupported runtime_t_d_fallback_mode: {runtime_t_d_fallback_mode}")

        self.gate = origin_mlp.gate
        self.base_t_d = self.num_experts // 2 if t_d is None else t_d
        self.t_d = self.base_t_d
        self.maxnnz = 4
        self.adaptive_t_d = adaptive_t_d
        self.bm_fallback_t_d = self.base_t_d if bm_fallback_t_d is None else bm_fallback_t_d
        self.forward_mode = forward_mode
        self.unsupported_batch_fallback_mode = unsupported_batch_fallback_mode
        self.runtime_t_d_fallback_mode = runtime_t_d_fallback_mode

        self.kernel_path_calls = 0
        self.unsupported_batch_fallback_calls = 0
        self.runtime_t_d_fallback_calls = 0
        self.original_forward_calls = 0
        self.bm_forward_calls = 0

    def get_path_stats(self):
        return {
            "kernel_path_calls": self.kernel_path_calls,
            "unsupported_batch_fallback_calls": self.unsupported_batch_fallback_calls,
            "runtime_t_d_fallback_calls": self.runtime_t_d_fallback_calls,
            "original_forward_calls": self.original_forward_calls,
            "bm_forward_calls": self.bm_forward_calls,
        }

    def _select_routing_inputs(self, router_logits):
        full_routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(full_routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        return routing_weights, selected_experts

    def _build_router_weights_from_topk(self, router_logits, routing_weights, selected_experts):
        router_weights = torch.zeros_like(router_logits)
        router_weights.scatter_(1, selected_experts, routing_weights)
        return router_weights

    def _build_routing_inputs(self, router_logits):
        routing_weights, selected_experts = self._select_routing_inputs(router_logits)
        router_weights = self._build_router_weights_from_topk(router_logits, routing_weights, selected_experts)
        return router_weights, routing_weights, selected_experts

    def mixer_with_fallback(self, router_logits):
        router_weights, routing_weights, selected_experts = self._build_routing_inputs(router_logits)
        expert_hit_count = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)
        col_sums = router_weights.sum(dim=0)
        runtime_t_d = torch.count_nonzero(expert_hit_count > self.maxnnz).item()
        self.t_d = runtime_t_d if self.adaptive_t_d else self.base_t_d
        effective_t_d = self.t_d if self.t_d > 0 else 1
        use_runtime_t_d_fallback = (
            self.runtime_t_d_fallback_mode != "none"
            and self.bm_fallback_t_d is not None
            and runtime_t_d > self.bm_fallback_t_d
        )

        sorted_experts = None
        mask_r = None
        if not use_runtime_t_d_fallback:
            _, sorted_experts = torch.sort(col_sums, descending=True)
            sparse_candidates = sorted_experts[-effective_t_d:]
            selected_weights = router_weights[:, sparse_candidates]
            _, mask_r = torch.topk(selected_weights, self.maxnnz, dim=0)

        return {
            "router_weights": router_weights,
            "routing_weights": routing_weights,
            "selected_experts": selected_experts,
            "sorted_experts": sorted_experts,
            "mask_r": mask_r,
            "col_sums": col_sums,
            "t_d": self.t_d,
            "runtime_t_d": runtime_t_d,
            "effective_t_d": effective_t_d,
            "use_runtime_t_d_fallback": use_runtime_t_d_fallback,
            "use_bm_fallback": use_runtime_t_d_fallback and self.runtime_t_d_fallback_mode == "bm",
        }

    def _run_fallback(
        self,
        mode: str,
        hidden_states: torch.Tensor,
        *,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        router_logits: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
        selected_experts: torch.Tensor | None = None,
    ):
        if mode == "bm":
            return self.bm_forward_with_extra_input(
                hidden_states,
                batch_size=batch_size,
                sequence_length=sequence_length,
                router_logits=router_logits,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
            )
        if mode == "original":
            if hidden_states.dim() == 2:
                if batch_size is None or sequence_length is None:
                    raise ValueError(
                        "batch_size and sequence_length are required for original fallback when hidden_states is 2D"
                    )
                hidden_states = hidden_states.view(batch_size, sequence_length, -1)
            return self.original_forward(hidden_states)
        raise ValueError(f"Unsupported fallback mode: {mode}")

    def original_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.original_forward_calls += 1
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = self._select_routing_inputs(router_logits)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            start_idx = expert_idx * self.intermediate_size
            end_idx = start_idx + self.intermediate_size
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            current_up = torch.matmul(current_state, self.combined_w3_weight[start_idx:end_idx, :].t())
            current_gate = torch.matmul(current_state, self.combined_w1_weight[start_idx:end_idx, :].t())
            current_activation = self.act_fn(current_gate)
            current_result = current_activation * current_up
            current_result = torch.matmul(current_result, self.combined_w2_weight[:, start_idx:end_idx].t())

            current_result = current_result * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_result.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def main_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        x = hidden_states.view(-1, hidden_dim)
        bs = x.size(0)

        if bs not in [32, 64, 128]:
            self.unsupported_batch_fallback_calls += 1
            return self._run_fallback(self.unsupported_batch_fallback_mode, hidden_states)

        router_logits = self.gate(x)
        mixer_output = self.mixer_with_fallback(router_logits)
        if mixer_output["use_runtime_t_d_fallback"]:
            self.runtime_t_d_fallback_calls += 1
            return self._run_fallback(
                self.runtime_t_d_fallback_mode,
                x,
                batch_size=batch_size,
                sequence_length=sequence_length,
                router_logits=router_logits,
                routing_weights=mixer_output["routing_weights"],
                selected_experts=mixer_output["selected_experts"],
            )

        self.kernel_path_calls += 1
        with torch.no_grad():
            router_weights = mixer_output["router_weights"]
            sorted_experts = mixer_output["sorted_experts"]
            mask_r = mixer_output["mask_r"]
            kernel_t_d = mixer_output["effective_t_d"]

            ir = torch.zeros((2, bs, self.intermediate_size * kernel_t_d), device=x.device, dtype=x.dtype)
            mask_v = torch.zeros((2, self.maxnnz, kernel_t_d * self.intermediate_size), device=x.device, dtype=x.dtype)
            result = torch.zeros((kernel_t_d, bs, self.hidden_size), device=x.device, dtype=x.dtype)

            mlp_kernel.ops.sddmm(
                x,
                self.combined_w3_weight,
                self.combined_w1_weight,
                ir,
                mask_r,
                sorted_experts,
                mask_v,
                bs,
                hidden_dim,
                self.total_intermediate_size,
                self.intermediate_size,
                kernel_t_d,
                self.maxnnz,
            )

            mlp_kernel.ops.spmm(
                ir[0],
                self.combined_w2_weight,
                result,
                mask_r,
                sorted_experts,
                mask_v,
                router_weights,
                bs,
                hidden_dim,
                self.total_intermediate_size,
                self.intermediate_size,
                kernel_t_d,
                self.maxnnz,
            )
            x = result.sum(0)

        hidden_states = x.reshape((batch_size, sequence_length, hidden_dim))
        return hidden_states, router_logits

    def bm_forward_with_extra_input(
        self,
        hidden_states: torch.Tensor,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        router_logits: torch.Tensor | None = None,
        routing_weights: torch.Tensor | None = None,
        selected_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.bm_forward_calls += 1
        if hidden_states.dim() == 3:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
        elif hidden_states.dim() == 2:
            hidden_dim = hidden_states.shape[-1]
            if batch_size is None or sequence_length is None:
                raise ValueError("batch_size and sequence_length are required when hidden_states is 2D")
        else:
            raise ValueError(f"Unsupported hidden_states shape: {hidden_states.shape}")

        if (routing_weights is None) != (selected_experts is None):
            raise ValueError("routing_weights and selected_experts must be provided together")

        if router_logits is None:
            router_logits = self.gate(hidden_states)

        if routing_weights is None:
            routing_weights, selected_experts = self._select_routing_inputs(router_logits)
        else:
            routing_weights = routing_weights.to(hidden_states.dtype)

        ffn_dim_per_expert = self.intermediate_size
        num_experts = self.num_experts
        bsl = batch_size * sequence_length

        expanded_hidden_states = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1)

        w1_weights = self.w1_weight_by_expert.transpose(-2, -1)
        w3_weights = self.w3_weight_by_expert.transpose(-2, -1)
        w2_weights = self.w2_weight_by_expert

        expert_inputs = torch.zeros(
            (num_experts, bsl, hidden_dim),
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

        token_indices = torch.arange(bsl, device=hidden_states.device).unsqueeze(1).expand(-1, self.top_k)
        final_expert_outputs_for_tokens = expert_outputs[selected_experts, token_indices]
        weighted_outputs = final_expert_outputs_for_tokens * routing_weights.unsqueeze(-1)
        final_hidden_states = weighted_outputs.sum(dim=1)

        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits

    def bm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.bm_forward_with_extra_input(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.forward_mode == "bm":
            return self.bm_forward(hidden_states)
        return self.main_forward(hidden_states)
