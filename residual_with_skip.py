"""
Residual Connections (Skip Connections)
Shows how skip connections help deep networks train better
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Plain deep network (no skip connections)
class PlainNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [784, 128, 128, 128, 128, 10]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ResNet-style network (with skip connections)
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
    
    def forward(self, x):
        return x + self.fc(x)  # Skip connection

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784, 128)
        self.blocks = nn.Sequential(*[ResBlock(128) for _ in range(3)])
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.blocks(x)
        return self.output(x)

# Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

# Train function
def train(model, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(epochs):
        for x, y in train_loader:
            pred = model(x)
            loss = nn.CrossEntropyLoss()(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses

# Train both models
print("Training Plain Network...")
plain_model = PlainNet()
plain_losses = train(plain_model)

print("Training ResNet...")
res_model = ResNet()
res_losses = train(res_model)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(plain_losses, label='Plain Network', alpha=0.7)
plt.plot(res_losses, label='ResNet (with skip)', alpha=0.7)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Residual Connections Help Training')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('residual_connections.png', dpi=150)
plt.show()

print(f"\nFinal Loss - Plain: {plain_losses[-100:] and sum(plain_losses[-100:])/100:.3f}")
print(f"Final Loss - ResNet: {sum(res_losses[-100:])/100:.3f}")
