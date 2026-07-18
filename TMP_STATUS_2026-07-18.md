# 临时状态记录（2026-07-18）

这个文件用于在中断后快速恢复当前上下文，不作为正式文档。

## 当前目标

先暂停 GPT-OSS / RedHat EAGLE3 对接排查，优先补 `Qwen3-30B-A3B` 的端到端 ablation。

## GPT-OSS / RedHat 当前状态

- 已保留的兼容修改：
  - `moe_test/EAGLE/eagle/model/ea_model.py`
    - 支持 `GptOssForCausalLM`
    - 支持从 `model.safetensors` 加载
    - 支持 `layers.0.* -> midlayer.*` 的 RedHat 权重名映射
    - 读取 `speculators_config` 并自动把 `speculative_tokens=3` 映射成当前运行时的 `total_token=4, depth=2, top_k=1`
    - 加载后根据 `t2d` 重建精确 `d2t`，修正了 3522 个错误 offset
  - `moe_test/EAGLE/eagle/model/configs.py`
  - `moe_test/EAGLE/eagle/model/cnets.py`
  - `moe_test/EAGLE/eagle/model/modeling_gpt_oss_kv.py`

- 已撤回的修改：
  - 曾尝试给 `ea_model.py` 增加一条单独的 greedy-chain runtime，但已回滚，不再保留。

## 最新验证结果

### RedHatAI/gpt-oss-20b-speculator.eagle3

旧结果（修精确 `d2t` 之前）：

- `1 prompt, max_new_tokens=128`
  - `Average time: 5.0335 s`
  - `Average AL: 0.7671`
  - 这是更早、未完全对齐的单次结果，参考意义有限
- `5 prompts, max_new_tokens=128`
  - `Average time: 4.0945 s`
  - `Average AL: 0.2995`

精确 `d2t` 修复之后：

- `1 prompt, max_new_tokens=128`
  - `Average time: 5.0394 s`
  - `Average AL: 0.4176`
- `5 prompts, max_new_tokens=128`
  - `Average time: 4.0998 s`
  - `Average AL: 0.3259`

结论：

- `d2t` 映射确实有 bug，修后 `AL` 有小幅提升。
- 但平均 `AL` 仍远低于预期，主问题大概率不在 tree-vs-chain runtime，而在 hidden-state 口径或更完整的 speculator runtime 对接。

## 下一步优先级

当前优先做：

- `Qwen3-30B-A3B`
- 真实 prompt 端到端 ablation
- `ea_total_token=64`（对应现在讨论的 `bs64` 口径）
- `maxnnz=4` 和 `maxnnz=8`
- 每组只跑 `tech1 / tech2 / tech3`
- 每组 `10 prompts`

tech 定义沿用当前 `moe_test/README.md`：

- `tech1`: `t_d=64`, `adaptive_t_d=False`, `runtime fallback=none`
- `tech2`: `t_d=64`, `adaptive_t_d=True`, `runtime fallback=none`
- `tech3`: `t_d=64`, `adaptive_t_d=True`, `runtime fallback=bm`

当前计划阈值：

- `maxnnz=4`: `bm_fallback_t_d=64`（已有控变量结果显示最优基本退化为“不触发 bm fallback”）
- `maxnnz=8`: `bm_fallback_t_d=40`（已有控变量结果中的最佳阈值）
