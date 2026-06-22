import torch

def unpack_tuple(x):
    if isinstance(x, (tuple, list)):
        return unpack_tuple(x[0])
    return x