import torch
import numpy as np
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_manual_seed(manual_seed):
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

def log_stats(losses, names, num_steps, writer):
    tag_value = {}
    for i in range(len(losses)):
        tag_value[f'loss {names[i]}'] = losses[i]

    for tag, value in tag_value.items():
        writer.add_scalar(tag, value, num_steps)
