import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import matplotlib.pyplot as plt


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        batch, _, seq_len, _ = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contigous()
        attn_out = attn_out.view(batch, seq_len, self.d_model)
        return self.W_o(attn_out), weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)

        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads,
                 d_ff, num_layers, num_classes,
                 max_seq_len=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.classifier(self.dropout(x))
        return x


torch.manual_seed(42)

model = MiniTransformer(
    vocab_size=1000,
    d_model=32,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    num_classes=2
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model built successfully")
print(f"total parameters: {total_params:,}")
print()
print(model)

vocab_size = 1000
seq_len = 20
n_samples = 500

X = torch.randint(0, vocab_size, (n_samples, seq_len))
y = torch.randint(0, 2, (n_samples,))

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test),
                         batch_size=32)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_acc = []

print("\nTraining...\n")

for epoch in range(1, 11):

    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == y_batch).sum().item()

    train_acc = correct / len(X_train)
    epoch_loss = total_loss / len(train_loader)
    train_losses.append(epoch_loss)
 model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            out = model(X_batch)
            correct += (out.argmax(1) == y_batch).sum().item()

    test_acc = correct / len(X_test)
    test_acc.append(test_acc)

    print(f"Epoch {epoch:2d} | "
          f"Loss: {epoch_loss:.4f} | "
          f"Train Acc: {train_acc:.2%} | "
          f"Test Acc: {test_acc:.2%}")
fig, axes = plt.subplots(1,2, figsize=(12,5))

axes[0].plot(train_losses, marker="o", color="coral")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[1].plot(test_accs, marker="s", color="teal")
axes[1].set_title("test accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()

    
        

