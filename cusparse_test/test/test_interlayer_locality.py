import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def gen_data():
    raw_data = torch.load("/root/projects/MMLU/data/mis-7b.pt", weights_only=True)  # 32 x N x 18432
    _, data_output = torch.split(raw_data, [4096, 14336], dim=2)  # 32 x N x 4096, 32 x N x 14336
    k = int(14336 * 0.2)
    topk_pos = torch.topk(data_output, k, dim=2)[1].view(-1)
    mask = torch.zeros_like(data_output)
    dim0_pos = torch.arange(32).unsqueeze(1).repeat(1, 50000 * k).view(-1)
    dim1_pos = torch.arange(50000).unsqueeze(0).unsqueeze(2).repeat(32, 1, k).view(-1)
    mask[dim0_pos, dim1_pos, topk_pos] = 1
    torch.save(mask, "/root/projects/MMLU/data/mis-7b-top20mask.pt")


def main():
    data = torch.load("/root/projects/MMLU/data/mis-7b-top20mask.pt", weights_only=True)  # 32 x N x 14336
    data_t = torch.transpose(data, 1, 2)  # 32 x 14336 x N
    train_data, test_data = torch.split(data_t, [45000, 5000], dim=2)  # 32 x 14336 x 45000, 32 x 14336 x 5000
    # gen map
    reorder_map = torch.zeros((32, 14336))  # 32 x 14336
    val = train_data[0].T.to('cuda')
    print("data load")
    for layer_idx in range(20, 21):
        print(f"layer {layer_idx}")
        vt = train_data[layer_idx].to('cuda')
        sim = (vt @ val + 1) / (vt.sum(dim=1, keepdim=True) + 1)
        # max_val = torch.max(sim, dim=1)[0]
        # min_val = torch.min(sim, dim=1)[0]
        # print(max_val.max(), max_val.min())
        # print(min_val.max(), min_val.min())
        # del_val = max_val - min_val
        # print(del_val.max(), del_val.min())
        print("sim gen")
        # greedy
        mvs = []
        for i in tqdm(range(14336)):
            mv, mp = torch.max(sim[i], 0)
            mvs.append(mv)
            sim[:, mp] = -1
        print(max(mvs), min(mvs))


if __name__ == "__main__":
    # gen_data()
    main()
