import torch
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=1)

print(layer.weight)
print(layer.bias)
x = torch.randn(5, 3)
output = layer(x)
print(output.shape)
