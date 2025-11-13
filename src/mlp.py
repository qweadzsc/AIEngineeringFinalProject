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
        # self.up_w = self.ob.up_proj.weight.data                     # 14336 x 4096
        # self.gate_w = self.ob.gate_proj.weight.data                 # 14336 x 4096
        # self.down_w = self.ob.down_proj.weight.data.T.contiguous()  # 14336 x 4096
        # self.down1 = self.ob.down_proj.weight.data[:, :5760].contiguous()
        # self.down2 = self.ob.down_proj.weight.data[:, 5760:].T.contiguous()
        # self.predictor = self.ob.predictor
        self.cuda_module = cuda_module

        self.t_dense = 5760
        self.t = 0
        self.num = 0
        # self.temp_w = self.predictor.fc2.weight.data[5760:].T.contiguous()

    def forward(self, x, before_norm):
        """
        x: bs x seq_l x 4096
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.act_fn(self.up_proj(x)))

        1. inter_r_1 = relu(up @ x)
        2. inter_r_2 = relu(gate @ x)
        3. inter_r_1 = inter_r_1 * inter_r_2
        4. result = down @ inter_r_1
        """
        return self.origin_forward(x, before_norm)
        # return self.test_forward(x, before_norm)
        # print_mem()

        bs, seq_l, _ = x.shape
        x = x.view(-1, 4096)
        before_norm = before_norm.view(-1, 4096)

        t_dense = self.t_dense

        mask = (self.predictor.fc1(before_norm) @ self.temp_w)
        # mask = self.predictor(before_norm)
        mask_dense_ref = torch.round(mask.sigmoid())
        sp = 0.8
        xbs = x.size(0)
        k = int(xbs * sp)
        _, top_indices = torch.topk(mask, k=k, dim=0)
        mask.zero_()
        col_indices = torch.arange(8576, device=mask.device).view(1, 8576).repeat((k, 1)).view(-1)
        mask[top_indices.view(-1), col_indices] = 1.0
        
        mask_in_use = mask
        # print(mask_in_use.mean().item(), end=" ")
        # print(mask_dense_ref.mean().item())

        ir1_r = self.ob.act_fn(self.ob.up_proj(x)[:, :t_dense])
        ir2_r = self.ob.act_fn(self.ob.gate_proj(x)[:, :t_dense])
        ir1_r *= ir2_r
        mask_r1 = self.ob.act_fn(self.ob.up_proj(x)[:, t_dense:])
        mask_r2 = self.ob.act_fn(self.ob.gate_proj(x)[:, t_dense:])
        mask_r1 *= mask_r2
        result_r1 = ir1_r @ self.down_w[:t_dense]
        result_r2 = (mask_r1 * mask_in_use) @ self.down_w[t_dense:]
        result_r1 += result_r2
        
        # mask_real = (mask_r1 > 0).to(torch.float16)
        # acc = 1 - (mask_in_use - mask_real).abs().mean()
        # rec = (mask_in_use * mask_real).mean() / mask_real.mean()
        # print("pred sp:", mask_in_use.mean())
        # print("real sp:", mask_real.mean())
        # print("acc:", acc)
        # print("rec:", rec)

        return result_r1.view(bs, seq_l, 4096)

    def test_forward(self, x, before_norm):
        bs, seq_l, _ = x.shape
        x_r = x
        before_norm = before_norm.view(-1, 4096)
        bs = 16
        x = x_r.view(-1, 4096)[:16].contiguous()
        before_norm = before_norm[:16]

        t_dense = 10240
        topk = 320
        
        start_ours = torch.cuda.Event(enable_timing=True)
        end_ours = torch.cuda.Event(enable_timing=True)
        start_original = torch.cuda.Event(enable_timing=True)
        end_original = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        
        start_original.record()
        ir1_r = self.ob.act_fn(self.ob.up_proj(x))
        ir2_r = self.ob.act_fn(self.ob.gate_proj(x))
        ir1_r *= ir2_r
        result_r = ir1_r @ self.down_w
        end_original.record()

        torch.cuda.synchronize()
        original_time = start_original.elapsed_time(end_original)

        mask_indices = torch.topk(
            self.predictor.relu(before_norm @ self.predictor.fc1.weight.data.T) @ \
            self.temp_w, k=topk, dim=1
        ).indices.view(-1).contiguous()
        start_ours.record()
        # ttt = self.predictor.relu(before_norm @ self.predictor.fc1.weight.data.T) @ \
        #     self.temp_w
        nnz = topk * bs
        mask_v = torch.zeros(nnz, device=x.device, dtype=x.dtype)
        mask_r = torch.arange(bs+1, dtype=torch.int64, device=x.device) * topk
        ir = torch.zeros((bs, t_dense), device=x.device, dtype=x.dtype)
        result = torch.zeros((bs*5, x.size(1)), device=x.device, dtype=x.dtype)
        self.cuda_module.torch_launch_mlp_kernel(x, self.up_w, self.gate_w, self.down1, self.down2,
                                                 mask_indices, mask_r, mask_v, ir, result,
                                                 x.size(0), x.size(1), self.up_w.size(0), t_dense)
        end_ours.record()
        
        torch.cuda.synchronize()
        ours_time = start_ours.elapsed_time(end_ours)

        print(original_time, ours_time)

        # exit(-3)
        
        return x_r

        # mask = torch.zeros((x.size(0), self.up_w.size(0)-t_dense), dtype=x.dtype, device=x.device)
        # ri = torch.arange(x.size(0), device=x.device).view(x.size(0), 1).repeat(1, topk).view(-1)
        # mask[ri, mask_indices.view(-1)] = 1
        #
        # ir1_r = self.ob.act_fn(self.ob.up_proj(x)[:, :t_dense])
        # ir2_r = self.ob.act_fn(self.ob.gate_proj(x)[:, :t_dense])
        # ir1_r *= ir2_r
        # mask_r1 = self.ob.up_proj(x)[:, t_dense:] * mask
        # mask_r2 = self.ob.gate_proj(x)[:, t_dense:] * mask
        # mask_r1 *= mask_r2
        # result_r1 = ir1_r @ self.down_w[:t_dense]
        # result_r2 = mask_r1 @ self.down_w[t_dense:]
        # result_r1 += result_r2
        #
        # return result_r1.view(bs, seq_l, -1)

    def origin_forward(self, x, before_norm):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        ir1 = self.ob.act_fn(self.ob.gate_proj(x))
        # end.record()
        ir2 = self.ob.act_fn(self.ob.up_proj(x))
        ir1 *= ir2
        result = self.ob.down_proj(ir1)
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # self.t += elapsed_time
        # self.num += x.size(1)
        # print(x.shape)
        # print(self.t)
        # print(self.t/self.num)
        return result
