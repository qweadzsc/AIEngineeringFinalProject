import torch
import time


def print_dif(x, y):
    assert x.shape == y.shape, f"{x.shape} {y.shape}"
    print((x-y).abs().max())


def print_mem():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1e6} MB")


class MLPCUDA(torch.nn.Module):

    def __init__(self, original_block, cuda_module):
        super().__init__()
        self.ob = original_block
        self.up_w = self.ob.up_proj.weight.data                     # 14336 x 4096
        self.gate_w = self.ob.gate_proj.weight.data                 # 14336 x 4096
        self.down_w = self.ob.down_proj.weight.data.T.contiguous()  # 14336 x 4096
        self.predictor = self.ob.predictor
        self.cuda_module = cuda_module

        self.t_dense = 10240

        # self.buffer = torch.zeros((5360, 4096), dtype=torch.int32, device=self.up_w.device)
        # self.mask_v2 = torch.zeros(4096*64*20, dtype=torch.float16, device=self.up_w.device)
        # self.result = torch.zeros((2 * 6400, 4096), dtype=torch.float16, device=self.up_w.device)
        # self.inter_r_1 = torch.zeros((6400, self.t_dense), dtype=torch.float16, device=self.up_w.device)
        # self.inter_r_2 = torch.zeros((6400, self.t_dense), dtype=torch.float16, device=self.up_w.device)

    def forward(self, x, before_norm):
        """
        x: bs x seq_l x 4096
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.act_fn(self.up_proj(x)))

        1. inter_r_1 = relu(up @ x)
        2. inter_r_2 = relu(gate @ x)
        3. inter_r_1 = inter_r_1 * inter_r_2
        4. result = down @ inter_r_1
        """
        # return self.test_forward(x, before_norm)
        # print_mem()

        bs, seq_l, _ = x.shape
        x = x.view(-1, 4096)

        t_dense = self.t_dense

        mask = self.predictor(before_norm)
        hard_mask = torch.round(mask)
        mask = mask + (hard_mask - mask).detach()
        mask = mask.view(-1, 14336)
        mask_sparse = mask[:, t_dense:].to_sparse_csr()
        mask_c = mask_sparse.col_indices()
        mask_r = mask_sparse.crow_indices()
        mask_v1 = mask_sparse.values()
        mask_v2 = torch.zeros_like(mask_v1)
        nnz = mask_sparse._nnz()
        # print(nnz, nnz / mask.size(0) / (mask.size(1) - t_dense), mask.shape)

        buffer = torch.zeros((5360, 4096), dtype=torch.int32, device=self.up_w.device)
        result = torch.zeros((2 * x.size(0), x.size(1)), dtype=x.dtype, device=x.device)
        inter_r_1 = torch.zeros((x.size(0), t_dense), dtype=x.dtype, device=x.device)
        inter_r_2 = torch.zeros((x.size(0), t_dense), dtype=x.dtype, device=x.device)

        # print_mem()

        # ir1_r = self.ob.act_fn(self.ob.up_proj(x)[:, :t_dense])
        # ir2_r = self.ob.act_fn(self.ob.gate_proj(x)[:, :t_dense])
        # ir1_r *= ir2_r
        # mask_r1 = self.ob.act_fn(self.ob.up_proj(x)[:, t_dense:] * mask[:, t_dense:])
        # mask_r2 = self.ob.act_fn(self.ob.gate_proj(x)[:, t_dense:] * mask[:, t_dense:])
        # mask_r1 *= mask_r2
        # result_r1 = ir1_r @ self.down_w[:t_dense]
        # result_r2 = mask_r1 @ self.down_w[t_dense:]
        # result_r1 += result_r2
        event_st = torch.cuda.Event(True)
        event_ed = torch.cuda.Event(True)
        event_st.record()
        temp = self.ob.act_fn(self.ob.up_proj(x))*self.ob.act_fn(self.ob.gate_proj(x))
        result_r = self.ob.down_proj(temp)
        # result_r_s = self.ob.down_proj(temp*mask)
        event_ed.record()
        event_ed.synchronize()
        print(event_st.elapsed_time(event_ed))
        # torch.cuda.synchronize()

        self.cuda_module.torch_launch_cutlass_mlp(x, self.up_w, self.gate_w, self.down_w,
                                                  mask_c, mask_r, mask_v1, mask_v2,
                                                  inter_r_1, inter_r_2, result, buffer,
                                                  x.size(0), x.size(1), self.up_w.size(0), t_dense, nnz)

        # print_dif(ir1_r, inter_r_1)
        # print_dif(ir2_r, inter_r_2)
        # print_dif(mask_r1, mask_sparse.to_dense())
        # # print_dif(mask_r2.to_sparse_csr().values(), mask_v2)
        # print_dif(result_r1, result[:x.size(0)])
        # print_dif(result_r2, result[x.size(0):])
        result = result[:x.size(0)]
        print_dif(result_r, result)
        # print_dif(result_r_s, result)
        # print_dif(result_r, result_r_s)
        # print("--------------------------------------------------")
        # exit(-1)
        return result.view(bs, seq_l, 4096)

    def test_forward(self, x, before_norm):
        bs, seq_l, _ = x.shape
        x = x.view(-1, 4096)
        before_norm = before_norm.view(-1, 4096)

        t_dense = 10240
        topk = 200

        mask_indices = torch.topk(self.predictor.relu(before_norm @ self.predictor.fc1.weight.data.T) @ self.predictor.fc2.weight.data[t_dense:].T, k=topk, dim=1).indices
        mask = torch.zeros((x.size(0), self.up_w.size(0)-t_dense), dtype=x.dtype, device=x.device)
        ri = torch.arange(x.size(0), device=x.device).view(x.size(0), 1).repeat(1, topk).view(-1)
        mask[ri, mask_indices.view(-1)] = 1

        ir1_r = self.ob.act_fn(self.ob.up_proj(x)[:, :t_dense])
        ir2_r = self.ob.act_fn(self.ob.gate_proj(x)[:, :t_dense])
        ir1_r *= ir2_r
        mask_r1 = self.ob.up_proj(x)[:, t_dense:] * mask
        mask_r2 = self.ob.gate_proj(x)[:, t_dense:] * mask
        mask_r1 *= mask_r2
        result_r1 = ir1_r @ self.down_w[:t_dense]
        result_r2 = mask_r1 @ self.down_w[t_dense:]
        result_r1 += result_r2
        
        return result_r1.view(bs, seq_l, -1)
