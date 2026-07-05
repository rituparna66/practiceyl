import torch
import numpy as np

torch.manual_seed(42)

X = torch.randn(1, 100)
y = 3 * X + 2 + torch.randn(100, 1) * 0.5

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.1
epochs = 100

for epoch in range(epochs):
    y_pred = w * X + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0:
        print(f"epoch: {epoch:3d} , Loss: {loss.item():.4f} , w: {w.item()}:.3f, b: {b.item():.3f}")

print(f"nFinal: w={w.item():.3f}, b={b.item():.3f}")
