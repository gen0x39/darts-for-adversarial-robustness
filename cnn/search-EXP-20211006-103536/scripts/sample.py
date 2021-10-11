import torch

def sum(a: torch.Tensor):
    return torch.sum(a)

x = [1,2,3]
sum(x)