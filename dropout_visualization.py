# dropout_visualization.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------------
# 1️⃣ Simple input
# ------------------------
torch.manual_seed(42)  # for reproducibility
input_tensor = torch.randn(1, 10)  # input with 10 neurons

# ------------------------
# 2️⃣ Fully connected layer
# ------------------------
fc = nn.Linear(10, 10)
dropout = nn.Dropout(p=0.5)  # 50% neurons will be dropped

# ------------------------
# 3️⃣ Forward pass without dropout (evaluation mode)
# ------------------------
fc.eval()
output_no_dropout = fc(input_tensor)
output_no_dropout = torch.sigmoid(output_no_dropout)  # normalize activations

# ------------------------
# 4️⃣ Forward pass with dropout (training mode)
# ------------------------
fc.train()
output_with_dropout = dropout(fc(input_tensor))
output_with_dropout = torch.sigmoid(output_with_dropout)

# ------------------------
# 5️⃣ Visualize
# ------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(10), output_no_dropout.detach().numpy()[0])
plt.title("Without Dropout (Eval Mode)")
plt.ylim(0, 1)
plt.xlabel("Neuron Index")
plt.ylabel("Activation Value")

plt.subplot(1, 2, 2)
plt.bar(range(10), output_with_dropout.detach().numpy()[0])
plt.title("With Dropout (Train Mode)")
plt.ylim(0, 1)
plt.xlabel("Neuron Index")
plt.ylabel("Activation Value")

plt.tight_layout()
plt.show()
