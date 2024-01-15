"""Common utils that can be shared across different tasks.
"""
import torch

def move_to_device(obj, device):
    """
    """
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple([move_to_device(v, device) for v in obj])
    elif isinstance(obj, set):
        return set([move_to_device(v, device) for v in obj])
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj