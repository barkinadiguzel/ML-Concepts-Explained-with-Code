import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ------------------------
# 1️⃣ Dataset and Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input size
    transforms.ToTensor()           # Convert images to PyTorch tensors
])

dataset = CIFAR10(root='./data', download=True, train=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------------
# 2️⃣ Select a small model
# ------------------------
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.eval()  # Set model to evaluation mode

# ------------------------
# 3️⃣ Get a single image
# ------------------------
img, label = next(iter(dataloader))
# img shape: [1, 3, 224, 224]

# ------------------------
# 4️⃣ Extract feature maps
# ------------------------
with torch.no_grad():
    features = model.features(img)  # feature maps shape: [1, C, H, W]

# ------------------------
# 5️⃣ Visualize the first feature map
# ------------------------
# Take the first channel
feat_map = features[0, 0].cpu().numpy()

plt.imshow(feat_map, cmap='viridis')
plt.title("Feature Map Channel 0")
plt.colorbar()
plt.show()
