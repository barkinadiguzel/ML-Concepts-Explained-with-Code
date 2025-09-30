"""
Mixup Augmentation
==================
Creates new training samples by mixing pairs of images.
Formula: mixed = lambda * img1 + (1-lambda) * img2
"""
 
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST
data = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())

# Get two different digits
img1 = [img for img, label in data if label == 3][0]
img2 = [img for img, label in data if label == 7][0]

# Mixup function
mixup = lambda img1, img2, lam: lam * img1 + (1 - lam) * img2

# Visualize
fig, axes = plt.subplots(1, 6, figsize=(12, 2))
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for ax, lam in zip(axes, lambdas):
    mixed = mixup(img1, img2, lam)
    ax.imshow(mixed.squeeze(), cmap='gray')
    ax.set_title(f'λ={lam}')
    ax.axis('off')

plt.suptitle('Mixup: Blending digit 7 and digit 3')
plt.tight_layout()
plt.savefig('mixup_augmentation.png', dpi=150)
plt.show()
