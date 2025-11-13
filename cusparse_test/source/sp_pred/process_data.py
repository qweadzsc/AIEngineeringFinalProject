import torch
from tqdm import tqdm


row_data = torch.zeros((32, 50000, 4096+14336+14336), dtype=torch.float16)
for i in tqdm(range(32)):
    row_data[i] = torch.load(f"/share/xujiaming/train_machine/yongkang/cusparse_test/data/gen_data/ts_7b/{i}.pt",
                             weights_only=True).to(torch.float16)
prenorm, inter_result, mask = torch.split_with_sizes(row_data, [4096, 14336, 14336], dim=2)

path = "/share/xujiaming/train_machine/yongkang/cusparse_test/data/gen_data/"
print(prenorm.shape)
torch.save(prenorm.contiguous(), path+"prenorm.pt")
print(inter_result.shape)
torch.save(inter_result.contiguous(), path+"inter_result.pt")
print(mask.shape)
torch.save(mask.contiguous(), path+"mask.pt")
