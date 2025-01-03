import random
import numpy as np
import torch

def seed_everything(g_cpu, SEED=13):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True 
    g_cpu.manual_seed(SEED)


def denormalize(tensor, mean, std):
    tensor_copy = tensor.clone().detach()  # make a copy of the tensor
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)  # denormalize
    return tensor_copy
