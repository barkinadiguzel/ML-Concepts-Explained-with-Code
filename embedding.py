#for github

words = ["king", "queen", "man", "woman", "apple", "orange", "dog", "cat"]

import torch
import torch.nn as nn

# vocab_size = kelime sayısı, embedding_dim = 2D görselleştirmek için 2
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=2)

# Kelimeleri indexle
word_to_idx = {w: i for i, w in enumerate(words)}
indices = torch.tensor([word_to_idx[w] for w in words])

# Embedding vektörlerini al
vectors = embedding(indices).detach().numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
for i, word in enumerate(words):
    x, y = vectors[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)
plt.title("2D Embedding Visualization")
plt.show()

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plot için

# Kelime listesi
words = ["king", "queen", "man", "woman", "apple", "orange", "dog", "cat"]

# 3 boyutlu embedding
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=3)

# Kelime → index
word_to_idx = {w: i for i, w in enumerate(words)}
indices = torch.tensor([word_to_idx[w] for w in words])

# Vektörleri al (detach + numpy)
vectors = embedding(indices).detach().numpy()

# 3D görselleştirme
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

for i, word in enumerate(words):
    x, y, z = vectors[i]
    ax.scatter(x, y, z)
    ax.text(x+0.01, y+0.01, z+0.01, word)

ax.set_title("3D Embedding Visualization")
plt.show()
