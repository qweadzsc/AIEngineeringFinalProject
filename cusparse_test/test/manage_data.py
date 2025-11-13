import torch
import numpy as np
import matplotlib.pyplot as plt


def and_sum(t, dim):
    result = t.sum(dim=dim)
    return torch.clamp(result, max=1.)


def xor_sim(t1, t2, dim):
    result = (t1+t2-1).abs()
    return result.mean(dim=dim)


def main():
    data = torch.load("/root/projects/MMLU/data/relu-7b.pt", weights_only=True)
    activation_thre_abs = data.abs().max(dim=3, keepdim=True)[0] / 100
    activation = (data > activation_thre_abs).to(torch.float32)[:, :, -100:]
    activation2 = torch.clamp(activation + activation[:, :1], max=1.)
    act_rate = np.array([and_sum(activation2[8, :, :i+1], dim=1).mean(dim=1).numpy() for i in range(100)])
    act_rate = torch.tensor(act_rate).T
    # plt.plot(act_rate[0])
    plt.plot(act_rate.mean(dim=0))
    k = torch.argmin(act_rate[1:, -1])
    plt.plot(act_rate[k+1])
    plt.savefig("/root/projects/MMLU/data/result/sim_layer/min2")


if __name__ == "__main__":
    main()
