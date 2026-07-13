class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build encoding matrix once
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Sine for even indices, cosine for odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Visualize it
import matplotlib.pyplot as plt

pe_module = PositionalEncoding(d_model=64)
pe_matrix = pe_module.pe[0].detach().numpy()

plt.figure(figsize=(12, 4))
plt.imshow(pe_matrix[:50, :], aspect="auto", cmap="RdBu")
plt.colorbar()
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.title("Positional Encoding — each row = one position's encoding")
plt.show()
