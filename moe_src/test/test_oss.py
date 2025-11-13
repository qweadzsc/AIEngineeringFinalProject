import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from transformers.activations import ACT2FN

import mlp_kernel


def print_dif(x, y, eps=1e-5, print_it=True):
    if x.shape != y.shape:
        raise ValueError(f"Tensors must have the same shape. Got {x.shape} and {y.shape}")
    if x.device != y.device:
        y = y.to(x.device)

    diff = torch.abs(x - y)
    different_mask = diff > eps

    total_elements = x.numel()
    different_count = torch.sum(different_mask).item()
    percentage = (different_count / total_elements) * 100
    max_diff = torch.max(diff).item()

    if (print_it):
        print(f"Different positions: {different_count}/{total_elements} ({percentage:.4f}%)")
        print(f"Maximum absolute difference: {max_diff:.6e}")
    
    return different_count, total_elements, max_diff


def print_dense_dif(x, y, mask_c, expert_w=768, eps=1e-5):
    diff = []
    total_elements = 0
    different_count = 0
    for i, c in enumerate(mask_c):
        dc, te, md = print_dif(x[:, i*expert_w:(i+1)*expert_w], y[:, c*expert_w:(c+1)*expert_w], eps=eps, print_it=False)
        diff.append(md)
        total_elements += te
        different_count += dc
    percentage = (different_count / total_elements) * 100
    max_diff = torch.max(torch.tensor(diff)).item()
    print(f"Different positions: {different_count}/{total_elements} ({percentage:.4f}%)")
    print(f"Maximum absolute difference: {max_diff:.6e}")


def print_select_dif(x, y, mask_r, mask_c, expert_w=768, eps=1e-5):
    mn = x.shape[0]
    sp = mask_c.shape[0]
    asp = y.shape[1] // expert_w
    x_reshaped = x.view(mn, sp, expert_w)
    y_reshaped = y.view(-1, asp, expert_w)
    sy = y_reshaped[:, mask_c, :]  # shape: [64, sp, expert_w]
    
    i_indices = torch.arange(mn)[:, None]  # shape: [mn, 1]
    j_indices = torch.arange(sp)[None, :]  # shape: [1, sp]

    x_selected = x_reshaped[i_indices, j_indices]  # shape: [mn, sp, expert_w]
    sy_selected = sy[mask_r, j_indices]            # shape: [mn, sp, expert_w]

    diff = torch.abs(x_selected - sy_selected)

    diff_count = torch.sum(diff > eps).item()
    max_diff = torch.max(diff).item()

    total_elements = x.numel()
    percentage = (diff_count / total_elements) * 100
    print(f"Different positions: {diff_count}/{total_elements} ({percentage:.4f}%)")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    
    return diff_count, max_diff


def set_all_seed(seed: int = 42, strict: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, hid_dim, inter_dim, bias=False, dtype=torch.float16):
        super().__init__()
        self.hid_dim = hid_dim
        self.inter_dim = inter_dim
        self.up_proj = nn.Linear(self.hid_dim, self.inter_dim, bias=bias,
                                 dtype=dtype, device='cuda')
        self.gate_proj = nn.Linear(self.hid_dim, self.inter_dim, bias=bias,
                                   dtype=dtype, device='cuda')
        self.down_proj = nn.Linear(self.inter_dim, self.hid_dim, bias=bias,
                                   dtype=dtype, device='cuda')
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class TestCUDAMoe(nn.Module):
    def __init__(self, hid_dim, t_d, maxnnz, cuda_module=None):
        super().__init__()
        self.num_experts = 32
        self.expert_w = 2944
        self.top_k = 4
        self.norm_topk_prob = True
        self.cuda_module = cuda_module
        self.hid_dim = hid_dim
        
        self.t_d = t_d
        self.maxnnz = maxnnz
        self.sp_pd = 1
        self.single_batch = 64

        self.gate = nn.Linear(hid_dim, self.num_experts, bias=False, dtype=torch.float16)
        self.experts = nn.ModuleList(
            [MLP(hid_dim, self.expert_w) for _ in range(self.num_experts)]
        )
        self.single = MLP(hid_dim, self.expert_w*self.num_experts, bias=True, dtype=torch.float16)
        self.u = self.single.up_proj.weight.data
        self.g = self.single.gate_proj.weight.data
        self.d = self.single.down_proj.weight.data
        self.ub = self.single.up_proj.bias.data
        self.gb = self.single.gate_proj.bias.data
        self.db = self.single.down_proj.bias.data
        self.alpha = 1.702
        self.limit = 7.0
        self.original_expert_dim = 2880
        
        self.data = []
    
    def rand_init(self):
        self.gate.weight.data = torch.rand_like(self.gate.weight) / 5
        for param in self.single.parameters():
            param.data = torch.rand_like(param) / 5
        for expert in self.experts:
            for param in expert.parameters():
                param.data = torch.rand_like(param) / 5
        self.u = self.single.up_proj.weight.data
        self.g = self.single.gate_proj.weight.data
        self.d = self.single.down_proj.weight.data
        self.ub = self.single.up_proj.bias.data
        self.gb = self.single.gate_proj.bias.data
        self.db = self.single.down_proj.bias.data
    
    def forward(self, hid):
        batch_size, sequence_length, hidden_dim = hid.shape
        x = hid.view(-1, hidden_dim)
        bs = x.size(0)
        
        x_ref = x.clone()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            router_logits = self.gate(x)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            values, indices = torch.topk(routing_weights, k=self.top_k, dim=1)
            values.div_(values.sum(1, keepdim=True))
            sparse_routing_weights = torch.zeros_like(routing_weights)
            sparse_routing_weights.scatter_(1, indices, values)
            routing_weights = sparse_routing_weights.to(x.dtype)

            # flat_indices = indices.view(-1)
            # counts = torch.bincount(flat_indices, minlength=routing_weights.size(1))
            # routing_n4 = (counts > 4).sum().item()
            # routing_n1 = (counts > 0).sum().item() // 2
            # t_d, maxnnz = max(routing_n1, routing_n4), 4

            routing_n = torch.count_nonzero(routing_weights, dim=0)
            t_d, maxnnz = self.t_d, self.maxnnz

            ir = torch.zeros(2, bs, t_d*self.expert_w, dtype=x.dtype, device=x.device)
            mask_c = torch.topk(routing_n, t_d // self.sp_pd + t_d).indices
            mask_r = sparse_routing_weights[:, mask_c[t_d:]].topk(maxnnz, dim=0).indices
            mask_v = torch.zeros(2, maxnnz, (t_d//self.sp_pd)*self.expert_w,
                                dtype=x.dtype, device=x.device)
            result = torch.zeros(t_d, bs, self.hid_dim, dtype=x.dtype, device=x.device)
            ub = self.ub.view(self.num_experts, self.expert_w)[mask_c]
            gb = self.gb.view(self.num_experts, self.expert_w)[mask_c]
        
            start_event.record()

            mlp_kernel.ops.sddmm(
                x, self.u, self.g, ir, mask_r, mask_c, mask_v,
                bs, hidden_dim, self.num_experts*self.expert_w, self.expert_w, t_d, maxnnz)
            
            ir[0] += ub[None, :t_d].view(1, -1)
            ir[1] += gb[None, :t_d].view(1, -1)
            mask_v[0] += ub[None, t_d:].view(1, -1)
            mask_v[1] += gb[None, t_d:].view(1, -1)

            ir[1] = ir[1].clamp(min=None, max=self.limit)
            ir[0] = ir[0].clamp(min=-self.limit, max=self.limit)
            mask_v[1] = mask_v[1].clamp(min=None, max=self.limit)
            mask_v[0] = mask_v[0].clamp(min=-self.limit, max=self.limit)

            gate_activ = ir[1] * torch.sigmoid(ir[1] * self.alpha)
            ir[0] = (ir[0] + 1) * gate_activ
            gate_activ_mask = mask_v[1] * torch.sigmoid(mask_v[1] * self.alpha)
            mask_v[0] = (mask_v[0] + 1) * gate_activ_mask
            ir.view(2, bs, t_d, self.expert_w)[0, :, :, self.original_expert_dim:] = 0
            mask_v.view(2, maxnnz, t_d, self.expert_w)[0, :, :, self.original_expert_dim:] = 0

            mlp_kernel.ops.spmm(
                ir[0], self.d, result, mask_r, mask_c, mask_v[0], routing_weights,
                bs, hidden_dim, self.num_experts*self.expert_w, self.expert_w, t_d, maxnnz)
            
            x = result.sum(0)
        
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        self.data.append(elapsed_time)
        if len(self.data) > 500 and (len(self.data) - 500) % 500 == 0:
            warmup_data = self.data[500:len(self.data)]
            avg_time = sum(warmup_data) / len(warmup_data)
            print(f"Average time for iterations {501}-{len(self.data)}: {avg_time:.4f} ms")
        
        x_ref = x_ref.repeat(self.num_experts, 1)
        x_ref = x_ref.view(self.num_experts, -1, hidden_dim)
        gate = torch.bmm(x_ref, self.g.view(self.num_experts, self.expert_w, hidden_dim).transpose(1, 2)) + self.gb.view(self.num_experts, self.expert_w)[..., None, :]
        up = torch.bmm(x_ref, self.u.view(self.num_experts, self.expert_w, hidden_dim).transpose(1, 2)) + self.ub.view(self.num_experts, self.expert_w)[..., None, :]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        activated_up = (up + 1) * glu
        activated_up[..., self.original_expert_dim:] = 0
        ref = activated_up.transpose(0, 1).reshape(bs, -1)
        print_dense_dif(ir[0], ref, mask_c[:self.t_d], self.expert_w, ref.abs().max()/200)
        print_select_dif(mask_v[0], ref, mask_r, mask_c[self.t_d:], self.expert_w, ref.abs().max()/200)

        ref = torch.zeros_like(x)
        for i, c in enumerate(mask_c[:self.t_d]):
            ref += (ir[0, :, i*self.expert_w:(i+1)*self.expert_w] @ self.d[:, c*self.expert_w:(c+1)*self.expert_w].T) * \
                routing_weights[:, c].unsqueeze(1)
        spd = self.d.view(2048, self.num_experts, self.expert_w)[:, mask_c[self.t_d:]]
        mask_v = mask_v.view(2, self.maxnnz, self.t_d//self.sp_pd, self.expert_w)
        for i in range(self.t_d//self.sp_pd):
        # for i in [0]:
            spr_ref = mask_v[0, :, i] @ spd[:, i].T
            for j in range(self.maxnnz):
                ref[mask_r[j, i]] += spr_ref[j] * \
                    routing_weights[mask_r[j, i], mask_c[self.t_d+i]]
        print_dif(x, ref, ref.abs().max()/200)

        return x.view(batch_size, sequence_length, hidden_dim)
    
    def dense_forward(self, x):
        batch_size, sequence_length, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        bs = x.size(0)
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=x.dtype)

        ir = x @ self.u[:32*768].T * self.single.act_fn(x @ self.g[:32*768].T)

        x = ir @ self.d.T[:32*768]
                
        return x
    
    def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.experts[expert_idx](current_state) * \
                                    routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits
        
    

if __name__ == "__main__":
    set_all_seed()
    # torch.set_printoptions(
    #     precision=2,        # 保留2位小数
    #     sci_mode=False,     # 不使用科学计数法
    # )
    hid_dim = 2048
    # cutlass_path = "/share/public/zhouyongkang/projects/sc/deps/cutlass"
    # dsmm_path = "/share/public/zhouyongkang/projects/sc/moe_src"
    # build_path = "/share/public/zhouyongkang/projects/sc/moe_src/build"
    # cuda_module = load(
    #     name="mlp_kernel",
    #     sources=[f.name for f in Path(dsmm_path).glob("*.cu")],
    #     extra_include_paths=[f'{cutlass_path}/include', dsmm_path],
    #     build_directory=build_path,
    #     verbose=True,
    #     extra_cuda_cflags=[
    #         '-arch=sm_80',
    #         '-O3',
    #         '--use_fast_math',
    #         '-Xptxas', '-v',
    #         '-g', '-lineinfo',
    #     ])
    t_d = 16
    maxnnz = 4
    for t_d in [16]:
    # for t_d in range(24, 40):
    # for maxnnz in range(1, 9):
        tc = TestCUDAMoe(hid_dim, t_d, maxnnz)
        # tc.test_forward = torch.compile(tc.forward)
        tc.rand_init()
        tc = tc.to('cuda')
        x = (torch.rand(1, 64, hid_dim, device='cuda', dtype=torch.float16)-0.5) / 2

        for _ in range(1):
            tc(x)
            # tc.test_forward(x)
            # tc.moe_forward(x)
            # tc.dense_forward(x)
        print('-'*50)
