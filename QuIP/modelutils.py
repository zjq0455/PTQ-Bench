import torch
import torch.nn as nn

DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Linear], name=''): # nn.Conv2d, 
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(child,
                        layers=layers,
                        name=name + '.' + name1 if name != '' else name1))
    return res
