import torch
MY_MLP_LIBRARY_NAME = "mlp_kernel"

__all__ = ["sddmm", "spmm"]

@torch.library.register_fake(f"{MY_MLP_LIBRARY_NAME}::launch_sddmm_kernel")
def _(x, up, gate, ir, mask_r, mask_c, mask_v, bs: int, in_d: int, mid_d: int, exp_d: int, t_dense: int, maxnnz: int):
    """Meta function for launch_sddmm_kernel."""
    torch._check(x.device == up.device == gate.device == ir.device == mask_r.device == mask_c.device == mask_v.device)
    torch._check(x.dtype == up.dtype == gate.dtype == ir.dtype == mask_v.dtype == torch.float16)
    torch._check(mask_r.dtype == mask_c.dtype == torch.long)


@torch.library.register_fake(f"{MY_MLP_LIBRARY_NAME}::launch_spmm_kernel")
def _(x, down, result, mask_r, mask_c, mask_v, exp_w, bs: int, in_d: int, mid_d: int, exp_d: int, t_dense: int, maxnnz: int):
    """Meta function for launch_spmm_kernel."""
    torch._check(x.device == down.device == result.device == mask_r.device == mask_c.device == mask_v.device == exp_w.device)
    torch._check(x.dtype == down.dtype == result.dtype == mask_v.dtype == exp_w.dtype == torch.float16)
    torch._check(mask_r.dtype == mask_c.dtype == torch.long)


def sddmm(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    ir: torch.Tensor,
    mask_r: torch.Tensor,
    mask_c: torch.Tensor,
    mask_v: torch.Tensor,
    bs: int,
    in_d: int,
    mid_d: int,
    exp_d: int,
    t_dense: int,
    maxnnz: int
) -> None:
    """
    User-friendly interface for the custom SDDMM kernel.

    Args:
        x (Tensor): Input tensor [...]
        up (Tensor): Input tensor [...]
        gate (Tensor): Input tensor [...]
        ir (Tensor): Input/Output tensor [...] (modified in-place)
        mask_r (Tensor): Input tensor (Long) [...]
        mask_c (Tensor): Input tensor (Long) [...]
        mask_v (Tensor): Input/Output tensor (Half) [...] (modified in-place)
        bs (int): Batch size
        in_d (int): Input dimension
        mid_d (int): Mid dimension
        exp_d (int): Expert dimension
        t_dense (int): T dense
        maxnnz (int): Max non-zero elements

    Returns:
        None: The operation modifies `ir` and `mask_v` tensors in-place.
    """
    torch.ops.mlp_kernel.launch_sddmm_kernel(
        x, up, gate, ir, mask_r, mask_c, mask_v,
        bs, in_d, mid_d, exp_d, t_dense, maxnnz
    )

def spmm(
    x: torch.Tensor,
    down: torch.Tensor,
    result: torch.Tensor,
    mask_r: torch.Tensor,
    mask_c: torch.Tensor,
    mask_v: torch.Tensor,
    exp_w: torch.Tensor,
    bs: int,
    in_d: int,
    mid_d: int,
    exp_d: int,
    t_dense: int,
    maxnnz: int
) -> None:
    """
    User-friendly interface for the custom SPMM kernel.

    Args:
        x (Tensor): Input tensor [...]
        down (Tensor): Input tensor [...]
        result (Tensor): Output tensor [...] (modified in-place)
        mask_r (Tensor): Input tensor (Long) [...]
        mask_c (Tensor): Input tensor (Long) [...]
        mask_v (Tensor): Input tensor (Half) [...]
        exp_w (Tensor): Input tensor (Half) [...]
        bs (int): Batch size
        in_d (int): Input dimension
        mid_d (int): Mid dimension
        exp_d (int): Expert dimension
        t_dense (int): T dense
        maxnnz (int): Max non-zero elements

    Returns:
        None: The operation modifies the `result` tensor in-place.
    """
    torch.ops.mlp_kernel.launch_spmm_kernel(
        x, down, result, mask_r, mask_c, mask_v, exp_w,
        bs, in_d, mid_d, exp_d, t_dense, maxnnz
    )
