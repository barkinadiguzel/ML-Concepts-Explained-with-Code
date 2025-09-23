# Word Embedding Visualization Example

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Word list
words = ["king", "queen", "man", "woman", "apple", "orange", "dog", "cat"]

# -------------------------------
# 2D Embedding
# -------------------------------
# Create embedding layer: vocab_size = number of words, embedding_dim = 2 for 2D visualization
embedding_2d = nn.Embedding(num_embeddings=len(words), embedding_dim=2)

# Map words to indices
word_to_idx = {w: i for i, w in enumerate(words)}
indices = torch.tensor([word_to_idx[w] for w in words])

# Get embedding vectors (detach from computation graph and convert to numpy)
vectors_2d = embedding_2d(indices).detach().numpy()

# Plot 2D embeddings
plt.figure(figsize=(6,6))
for i, word in enumerate(words):
    x, y = vectors_2d[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)
plt.title("2D Embedding Visualization")
plt.show()

# -------------------------------
# 3D Embedding
# -------------------------------
# Create 3D embedding layer
embedding_3d = nn.Embedding(num_embeddings=len(words), embedding_dim=3)

# Get embedding vectors
vectors_3d = embedding_3d(indices).detach().numpy()

# Plot 3D embeddings
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i, word in enumerate(words):
    x, y, z = vectors_3d[i]
    ax.scatter(x, y, z)
    ax.text(x+0.01, y+0.01, z+0.01, word)

ax.set_title("3D Embedding Visualization")
plt.show()
