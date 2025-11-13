import torch
import time
import matplotlib.pyplot as plt
import transformers


def time_dense(m, n, k, n_run=100, n_warmup=10):
    A = torch.rand(m, k)
    B = torch.rand(k, n)
    A = A.to('cuda:0')
    B = B.to('cuda:0')

    for _ in range(n_warmup):
        torch.matmul(A, B)
        torch.cuda.synchronize()

    total_time = 0
    for _ in range(n_run):
        start = time.time()
        result = torch.matmul(A, B)
        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start

    avg_time = total_time / n_run
    print(f"dense {m} x {k} x {n}: {avg_time*1000:.4f}ms")
    return avg_time


def time_sddmm(m, n, k, s, n_run=100, n_warmup=10):
    A = torch.rand(m, k)
    B = torch.rand(k, n)
    A = A.to('cuda:0')
    B = B.to('cuda:0')

    nz = int(s * m * n)
    # crow_indices = torch.randint(0, nz + 1, (m + 1,))
    # crow_indices = torch.sort(crow_indices)[0]
    # crow_indices[0] = 0
    # crow_indices[-1] = nz
    crow_indices = torch.arange(m+1) / m * nz
    crow_indices = crow_indices.to(torch.int)
    # col_indices = torch.randint(0, n, (nz,))
    # col_indices[nz//2] = n-1
    col_indices = torch.zeros((nz,), dtype=torch.int)
    for i in range(m):
        col_indices[crow_indices[i]:crow_indices[i+1]] = torch.randperm(n)[:crow_indices[i+1]-crow_indices[i]]
    if col_indices.max() < n-1:
        col_indices[nz//2] = n-1
    values = torch.ones(nz)

    S = torch.sparse_csr_tensor(crow_indices, col_indices, values).to('cuda:0')

    for _ in range(n_warmup):
        torch.sparse.sampled_addmm(S, A, B, beta=0)
        torch.cuda.synchronize()

    total_time = 0
    for _ in range(n_run):
        start = time.time()
        result = torch.sparse.sampled_addmm(S, A, B, beta=0)
        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start

    avg_time = total_time / n_run
    print(f"sddmm {nz} - {m} x {k} x {n}: {avg_time*1000:.4f}ms")
    return avg_time


def multiple_timer(func, n=10):
    total_time = 0
    for _ in range(n):
        time_u = func()
        total_time += time_u
    return total_time / n


def time_and_draw(m, n, k, s_list):
    for s in s_list:
        max_times = []
        for i in range(1, 50):
            n1 = int(n * i / 50)
            n2 = n - n1
            time_dense_result = multiple_timer(lambda : time_dense(m, n1, k))
            ss = (1-9*(i-1)/490) * s
            time_sddmm_result = multiple_timer(lambda : time_sddmm(m, n2, k, ss))
            max_time = max(time_dense_result, time_sddmm_result)
            max_times.append(max_time)

        plt.plot(range(1, 50), max_times, label=f's={s}')

    plt.legend()
    plt.savefig(f'/root/projects/MMLU/data/result/sddmm_time/{m}_{n}.png')
    plt.clf()


if __name__ == "__main__":
    m = 4
    n = 14336
    k = 4096
    s_list = [0.02, 0.05, 0.1, 0.15]
    time_and_draw(m, n, k, s_list)
