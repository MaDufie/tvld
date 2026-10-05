import torch


def scatter(src, index, dim=0, dim_size=None, reduce="sum", out=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    idx_expand = index
    if src.dim() > 1:
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        idx_expand = index.view(view_shape).expand_as(src)
    if reduce == "sum" or reduce == "add":
        out_ = torch.zeros(shape, dtype=src.dtype, device=src.device)
        out_.scatter_add_(dim, idx_expand, src)
        return out_
    elif reduce == "max" or reduce == "amax":
        out_ = torch.zeros(shape, dtype=src.dtype, device=src.device)
        out_.scatter_reduce_(dim, idx_expand, src, reduce="amax", include_self=False)
        return out_
    else:
        raise NotImplementedError(f"scatter shim: reduce={reduce!r} not implemented")
