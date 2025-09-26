import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
 
# ------------------------
# 1️⃣ Dataset and Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Use only 2 classes for simplicity: airplane (0) and automobile (1)
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
idx = torch.tensor([i for i, (_, label) in enumerate(dataset) if label in [0,1]])
subset = torch.utils.data.Subset(dataset, idx)
dataloader = DataLoader(subset, batch_size=32, shuffle=True)

# ------------------------
# 2️⃣ Simple MLP Models
# ------------------------
class SimpleMLP(nn.Module):
    def __init__(self, use_layernorm=False):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*32*3, 128)
        self.ln = nn.LayerNorm(128) if use_layernorm else nn.Identity()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # 2 classes

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Two models: one with LayerNorm, one without
model_ln = SimpleMLP(use_layernorm=True)
model_no_ln = SimpleMLP(use_layernorm=False)

# ------------------------
# 3️⃣ Training Setup
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer_ln = optim.Adam(model_ln.parameters(), lr=0.001)
optimizer_no_ln = optim.Adam(model_no_ln.parameters(), lr=0.001)

def train(model, optimizer, dataloader, epochs=5):
    acc_list = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        acc_list.append(acc)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")
    return acc_list

# ------------------------
# 4️⃣ Train both models
# ------------------------
print("Training Model with LayerNorm")
acc_ln = train(model_ln, optimizer_ln, dataloader)

print("\nTraining Model without LayerNorm")
acc_no_ln = train(model_no_ln, optimizer_no_ln, dataloader)

# ------------------------
# 5️⃣ Visualize Accuracy Difference
# ------------------------
plt.plot(acc_ln, label="With LayerNorm")
plt.plot(acc_no_ln, label="Without LayerNorm")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Effect of LayerNorm on Training")
plt.legend()
plt.show()
