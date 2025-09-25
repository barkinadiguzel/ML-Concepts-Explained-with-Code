#https://poloclub.github.io/cnn-explainer  you can look this side remember "visualize visualize visualize"
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# --- Mini grayscale image (8x8) ---
image = torch.tensor([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0]
], dtype=torch.float32)

image = image.unsqueeze(0).unsqueeze(0)  # shape: [batch, channel, H, W]

# --- Define simple Conv layer ---
conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
torch.manual_seed(0)  # reproducibility
nn.init.kaiming_normal_(conv.weight)  # He initialization for conv
nn.init.zeros_(conv.bias)

# --- Apply convolution ---
with torch.no_grad():
    feature_maps = conv(image)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(9,3))

# Original image
axes[0].imshow(image[0,0], cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

# Feature map 1
axes[1].imshow(feature_maps[0,0], cmap='gray')
axes[1].set_title("Filter 1")
axes[1].axis('off')

# Feature map 2
axes[2].imshow(feature_maps[0,1], cmap='gray')
axes[2].set_title("Filter 2")
axes[2].axis('off')

plt.suptitle("Convolution Filters Demo")
plt.show()
