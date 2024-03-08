import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F



def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
       
        if 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
                #print("paras[2]")
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
            #print("recursive")
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
                #print("paras[1]")
    return paras

