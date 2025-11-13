import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time


def box_fig():
    """
    real_mask = torch.stack(real_mask_ds, dim=0).numpy()
    pred_mask = torch.stack(real_mask_sp, dim=0).numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = np.arange(1, 33)
    width = 0.35

    for i, pos in enumerate(positions):
        bp_real = ax.boxplot([real_mask[i]], positions=[pos - width/2], widths=width,
                             patch_artist=True, showfliers=False)
        for patch in bp_real['boxes']:
            patch.set_facecolor('blue')
        bp_pred = ax.boxplot([pred_mask[i]], positions=[pos + width/2], widths=width,
                             patch_artist=True, showfliers=False)
        for patch in bp_pred['boxes']:
            patch.set_facecolor('orange')

        real_min = np.min(real_mask[i])
        real_max = np.max(real_mask[i])
        ax.plot([pos - width/2, pos - width/2], [real_min, real_max], color='black', linewidth=1)
        ax.plot([pos - width/2 - width/4, pos - width/2 + width/4], [real_min, real_min],
                color='black', linewidth=1)
        ax.plot([pos - width/2 - width/4, pos - width/2 + width/4], [real_max, real_max],
                color='black', linewidth=1)

        pred_min = np.min(pred_mask[i])
        pred_max = np.max(pred_mask[i])
        ax.plot([pos + width/2, pos + width/2], [pred_min, pred_max], color='black', linewidth=1)
        ax.plot([pos + width/2 - width/4, pos + width/2 + width/4], [pred_min, pred_min],
                color='black', linewidth=1)
        ax.plot([pos + width/2 - width/4, pos + width/2 + width/4], [pred_max, pred_max],
                color='black', linewidth=1)

    ax.set_title('Boxplot Comparison of real_mask and pred_mask')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Value')
    ax.set_xticks(positions)
    ax.set_xticklabels(np.arange(1, 33))

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='real_mask'),
                    Patch(facecolor='orange', label='pred_mask')]
    ax.legend(handles=legend_elements)
    """


import torch
import matplotlib.pyplot as plt
import numpy as np

def calculate_activation_stats_gpu(real_mask: torch.Tensor, k: int = 10752) -> None:
    device = real_mask.device
    batch_size, num_neurons, num_features = real_mask.shape

    stats = {'max': [], 'min': [], 'mean': []}

    real_mask = real_mask.to(device)
    act_i = real_mask.mean(dim=1)

    for layer_idx in range(batch_size):
        mask_i = real_mask[layer_idx]
        current_act = act_i[layer_idx]
        _, sparse_indices = torch.topk(current_act, k=num_features-k, largest=False)

        mask_sp_i = mask_i.index_select(dim=1, index=sparse_indices)
        # b = 8
        # mask_sp_proc = mask_sp_i.view(500, 100, 8576//b, b)
        act_sp_i = mask_sp_i.mean(dim=0)

        sp_max = act_sp_i.max().cpu().item()
        sp_min = act_sp_i.min().cpu().item()
        sp_mean = act_sp_i.mean().cpu().item()
        stats['max'].append(sp_max)
        stats['min'].append(sp_min)
        stats['mean'].append(sp_mean)

    plt.figure(figsize=(12, 6))
    layers = np.arange(1, batch_size + 1)

    plt.plot(layers, stats['max'], 'r-',  label='Sparse Max', linewidth=2, alpha=0.8)
    plt.plot(layers, stats['min'], 'r--', label='Sparse Min', linewidth=1.5, alpha=0.6)
    plt.plot(layers, stats['mean'], 'r.-', label='Sparse Mean', linewidth=1.2, alpha=0.4)

    y_ticks = np.arange(0, 0.11, 0.005)
    for y in y_ticks:
        plt.axhline(y, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Activation Rate')
    plt.title('Dense/Sparse Activation Rate Statistics')
    plt.xticks(layers[::2])  # 每两层显示刻度
    plt.ylim(0, 0.1)
    plt.grid(alpha=0.1)
    # plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    fig_path = "/share/public/zhouyongkang/projects/sc/cusparse_test/result/fig"
    
    real_mask_mmlu = torch.load(
        "/share/public/zhouyongkang/projects/sc/data/gen_data/inter_result_mmlu.pt",
        weights_only=True)
    real_mask_mmlu = (real_mask_mmlu > 0).to(torch.float16) #.to("cuda:5")
    # pred_mask_mmlu = torch.load(
    #     "/share/public/zhouyongkang/projects/sc/data/gen_data/mask_mmlu.pt",
    #     weights_only=True)
    # pred_mask_mmlu = (pred_mask_mmlu > 0).to(torch.float16).to("cuda:6")

    # calculate_activation_stats_gpu(real_mask_mmlu)
    aaa = real_mask_mmlu[0].mean(dim=1)
    plt.hist(aaa, 320, (0, 0.16))

    plt.savefig(f"{fig_path}/figmmlu_for_pre.png")
        