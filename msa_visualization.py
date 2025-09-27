# msa_visualization_safe.py
 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------------
# 1️⃣ Parameters
# ------------------------
seq_len = 5
embed_dim = 8
num_heads = 2

# ------------------------
# 2️⃣ Random token embeddings
# ------------------------
torch.manual_seed(42)
tokens = torch.randn(1, seq_len, embed_dim)  # [batch, seq_len, embed_dim]

# ------------------------
# 3️⃣ Multi-Head Self-Attention Layer
# ------------------------
msa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
msa.eval()
attn_output, attn_weights = msa(tokens, tokens, tokens)

# ------------------------
# 4️⃣ Convert attention weights to 2D
# ------------------------
# attn_weights shape: [batch, num_heads, seq_len, seq_len] veya [num_heads, seq_len, seq_len]
if attn_weights.dim() == 4:
    attn_matrix = attn_weights[0, 0].detach().cpu().numpy()
else:
    attn_matrix = attn_weights[0].detach().cpu().numpy()

# Ensure it's 2D
attn_matrix = attn_matrix.reshape(seq_len, seq_len)

# ------------------------
# 5️⃣ Visualize
# ------------------------
plt.imshow(attn_matrix, cmap='viridis')
plt.colorbar()
plt.title("Attention Weights - Head 0")
plt.xlabel("Key Token Index")
plt.ylabel("Query Token Index")
plt.show()
