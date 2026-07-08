import torch
import torch.nn as nn

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1
)

x = torch.randn(4, 1, 28, 28)
out = conv(x)
print(out.shape)
