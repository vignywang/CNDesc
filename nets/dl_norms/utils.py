import torch


def allclose_or_none(a, b, atol=1e-8):
    return (a is None and b is None) or torch.allclose(a, b, atol=atol)
