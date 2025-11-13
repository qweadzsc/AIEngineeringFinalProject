import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


input_sparsity = 0.4
output_sparsity = 0.85
input_size = 4096
outout_size = 14336

# data gen
def data_gen():
    raw_data = torch.load("/root/projects/MMLU/data/mis-7b.pt", weights_only=True)  # 32 x N x 18432
    data_input, data_output = torch.split(raw_data, [4096, 14336], dim=2)  # 32 x N x 4096, 32 x N x 14336
    # data_input_abs_thres = data_input.abs().max(dim=2, keepdim=True)[0] / 100  # 32 x N x 1
    # data_output_abs_thres = data_output.abs().max(dim=2, keepdim=True)[0] / 100  # 32 x N x 1
    # mask_input = (data_input > data_input_abs_thres).to(torch.float16)
    mask_output = (data_output > 1e-5).to(torch.float16)
    torch.save(mask_output, "/root/projects/MMLU/data/mis-7b-mask-out-ex.pt")
    # input_topk = int(input_size * (1-input_sparsity))
    # output_topk = int(outout_size * (1-output_sparsity))
    # rows_input = torch.arange(data_input.size(1)).unsqueeze(1).unsqueeze(0).repeat(data_input.size(0), 1, int(input_topk))  # 32 x N x in_k
    # rows_output = torch.arange(data_output.size(1)).unsqueeze(1).unsqueeze(0).repeat(data_output.size(0), 1, int(output_topk))  # 32 x N x out_k
    # layer_input = torch.arange(data_input.size(0)).unsqueeze(1).repeat(1, data_input.size(1)*input_topk)  # 32 x in_kN
    # layer_output = torch.arange(data_output.size(0)).unsqueeze(1).repeat(1, data_output.size(1)*output_topk)  # 32 x out_kN
    # data_input_mask_indice = torch.topk(data_input, k=input_topk, dim=2)[1]  # 32 x N x in_k
    # data_output_mask_indice = torch.topk(data_output, k=output_topk, dim=2)[1]  # 32 x N x out_k
    # data_input_mask = torch.zeros_like(data_input)
    # data_output_mask = torch.zeros_like(data_output)
    # data_input_mask[layer_input.view(-1), rows_input.view(-1), data_input_mask_indice.view(-1)] = 1
    # data_output_mask[layer_output.view(-1), rows_output.view(-1), data_output_mask_indice.view(-1)] = 1
    # torch.save(torch.cat([data_input_mask, data_output_mask], dim=2), "/root/projects/MMLU/data/mis-7b-mask.pt")


class Predictor(nn.Module):
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.fc1 = nn.Linear(in_c, hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hid, out_c)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.sigmoid()
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)
    

# class PredictorTrainer(nn.Module):
#     def __init__(self, n_layer, in_c, hid, out_c):
#         super().__init__()
#         self.predictors = nn.ModuleList([Predictor(in_c, hid, out_c) for _ in range(n_layer)])
#         self.n_layer = n_layer
#
#     def forward(self, x):  # 32 x B x 4096
#         return torch.cat([self.predictors[i](x[i]).unsqueeze(0) for i in range(self.n_layer)], dim=0)


def train(data, mask, i, config):
    """
    train helper

    data: N x 4096
    mask: N x 14336
    i: the layer idx

    return: (state_dict, acc, nzacc)
    """
    split = int(data.size(0) * config["split_rate"])
    dt = data[split:]  # dil train
    dv = data[:split]  # dil val
    mt = mask[split:]
    mv = mask[:split]
    bs = config["batch_size"]
    predictor = Predictor(input_size, config["hid_dim"], outout_size).to('cuda')
    predictor.apply(weights_init)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["gamma"])
    loss_fn = lambda x, y, alpha: -(alpha * y * torch.log(x+1e-5) + (1-alpha) * (1-y) * torch.log(1-x+1e-5)).mean()  # focal loss
    max_acc = 0
    epoch = 0
    count = 0
    psd = {}
    while True:
        predictor.train()
        for j in range(int((data.size(0) - split) / bs)):
            dtb = dt[j*bs:(j+1)*bs].to('cuda')
            mtb = mt[j*bs:(j+1)*bs].to('cuda')
            dtbh = predictor(dtb)
            loss = loss_fn(dtbh, mtb, torch.tanh(config["alpha"]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        predictor.eval()
        cor = 0
        num = 0
        nz = 0
        nzp = 0
        for j in range(split):
            dvj = dv[j].to('cuda')
            mvj = mv[j]
            dvjh = predictor(dvj).to('cpu')
            num += mvj.size(0)
            cor += torch.sum(mvj==torch.round(dvjh))
            nz += mvj.sum()
            nzp += torch.round(dvjh).sum()
        acc = cor/num
        nzr = nz/num
        nzpr = nzp/num
        nzacc = 0.5+(cor+nzp-num)/nz/2
        # print(f"epoch {epoch}, predictor {i}, acc {acc}, non-zero {nzr}, predict non-zero {nzpr}, non-zero acc {nzacc}, alpha {config['alpha']}")
        with torch.no_grad():
            config["alpha"] += (0.5*nzr-nzpr+0.2)*2

        if nzacc > max_acc:
            max_acc = nzacc
            psd = predictor.state_dict()
            count = 0
        else:
            count += 1
            if count == 5:
                break

        epoch += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch}, predictor {i}, acc {acc}, non-zero {nzr}, predict non-zero {nzpr}, non-zero acc {nzacc}, alpha {config['alpha']}")
        
    assert len(psd) > 0
    print(f"on break: epoch {epoch}, predictor {i}, acc {acc}, non-zero {nzr}, predict non-zero {nzpr}, non-zero acc {nzacc}, alpha {config['alpha']}")
    return psd, acc, nzacc


def main():
    # load data
    data = torch.load("/root/projects/MMLU/data/mis-7b.pt", weights_only=True)  # 32 x N x 18432
    mask = torch.load("/root/projects/MMLU/data/mis-7b-mask-out-ex.pt", weights_only=True)  # 32 x N x 14336
    mask_out = mask.to(torch.float32)  # 32 x N x 14336
    data_in = data[:, :, :4096].to(torch.float32)  # 32 x N x 4096
    need_resize = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2]
    data_dict = {}
    acc = []
    nzacc = []
    for i in range(mask_out.size(0)):
        # train config
        train_config = {
            "split_rate" : 0.1,  # test set in data
            "batch_size": 500,
            "hid_dim": 1024 + 512 * need_resize[i],
            "lr" : 1e-3,
            "gamma" : 0.999,  # gamma of scheduler
            "alpha" : torch.ones(1, device='cuda'),  # alpha of focal loss
        }
        start = time.time()
        state_dict, accuracy, nz_accuracy = train(data_in[i], mask_out[i], i, train_config)
        print(f"------------------------------------------------\ntime used in train predictor {i}: {time.time()-start}s\n------------------------------------------------")
        acc.append(accuracy)
        nzacc.append(nz_accuracy)
        data_dict[f"model.layers.{i}.predictor.fc1.weight"] = state_dict["fc1.weight"]
        data_dict[f"model.layers.{i}.predictor.fc1.bias"] = state_dict["fc1.bias"]
        data_dict[f"model.layers.{i}.predictor.fc2.weight"] = state_dict["fc2.weight"]
        data_dict[f"model.layers.{i}.predictor.fc2.bias"] = state_dict["fc2.bias"]
    torch.save(data_dict, "/root/projects/MMLU/data/mis-7b-predictor.pt")
    print(acc)
    print(nzacc)


if __name__ == "__main__":
    # data_gen()
    # main()
    # print(1-torch.tensor([0.9332, 0.9355, 0.9177, 0.9240, 0.9174, 0.9160, 0.8924, 0.8952, 0.8738,
    #     0.8463, 0.8293, 0.8162, 0.8188, 0.8171, 0.8259, 0.8485, 0.8647, 0.8665,
    #     0.8781, 0.8975, 0.9157, 0.9325, 0.9281, 0.9296, 0.9331, 0.9327, 0.9282,
    #     0.9305, 0.9306, 0.9123, 0.8923, 0.8814]))
    mask = torch.load("/root/projects/MMLU/data/mis-7b-mask-out-ex.pt", weights_only=True)  # 32 x N x 14336
    act_mean = mask.mean(dim=1)
    sorted_pos = act_mean.sort(dim=1)[1]
    for i in range(32):
        t = []
        for j in range(9):
            k = int(14336*(j+1)/10)
            selected_sum = mask[i, :, sorted_pos[i, :k]].sum(dim=1)
            t.append((selected_sum.max()-selected_sum.min())/k)
            if ((selected_sum.max()-selected_sum.min())/k) > 0.2:
                print(f"layer {i}: {j+1}")
        plt.plot(t)
    plt.savefig("/root/projects/MMLU/data/result/balance/t.png")
