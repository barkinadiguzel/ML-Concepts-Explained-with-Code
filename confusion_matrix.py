import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Mini dataset (3 classes) ---
# 3 classes, 2 features
X = torch.tensor([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0],
    [1.2, 0.8], [0.8, 1.5], [6.2, 8.5], [5.5, 9.2],
    [0.5, 0.7], [6.0, 8.0]
])
y = torch.tensor([0,0,1,1,0,0,1,1,2,2])  # 3 classes: 0,1,2

# --- Simple MLP model ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2,3)  # input=2, output=3 (3 classes)

    def forward(self, x):
        return self.fc(x)

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# --- Training loop ---
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# --- Predictions ---
with torch.no_grad():
    outputs = model(X)
    _, preds = torch.max(outputs, 1)

# --- Confusion Matrix ---
num_classes = 3
conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int32)

# Count predictions vs actual labels
for true_label in y:
    for pred_label in preds:
        pass  # we will fix below

for t, p in zip(y, preds):
    conf_mat[t, p] += 1

# --- Accuracy calculation ---
accuracy = torch.sum(torch.diag(conf_mat)).item() / torch.sum(conf_mat).item()
print(f"Accuracy: {accuracy*100:.2f}%")

# --- Visualization ---
plt.figure(figsize=(5,4))
sns.heatmap(conf_mat.numpy(), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
