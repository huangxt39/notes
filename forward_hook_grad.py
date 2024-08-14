# this experiment shows operations in forward_hook are recorded in autograd graph

import torch

l = torch.nn.Linear(3,4)

def hook(module, args, output):
    return 2 * output

    # output = output.detach().clone()
    # output.requires_grad_(True)
    # return output
    
l.register_forward_hook(hook)

x = torch.arange(3).unsqueeze(0).float()
y = l(x)

y.sum().backward()
print(l.weight.grad)
