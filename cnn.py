import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (batch, 32, 7, 7) -> (batch, 32*7*7)
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.fetaures(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using:{device}")

model = CNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            total_loss += criterion(output, y_batch).item()
            correct += (output.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

epochs = 5
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss,  test_acc  = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch}/{epochs} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2%}")
model.eval()
examples = next(iter(test_loader))
images, labels = examples
images, labels = images[:8].to(device), labels[:8].to(device)

with torch.no_grad():
    outputs = model(images)
    preds   = outputs.argmax(1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = images[i].cpu().squeeze().numpy()
    ax.imshow(img, cmap="gray")
    color = "green" if preds[i] == labels[i] else "red"
    ax.set_title(f"Pred: {preds[i].item()} | True: {labels[i].item()}",
                 color=color)
    ax.axis("off")

plt.suptitle("CNN Predictions on MNIST")
plt.tight_layout()
plt.show()    
