import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- Simple dataset (CIFAR10, train only) ---
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Get first image
images, labels = zip(*[(img, lbl) for img, lbl in [trainset[0]]])

# --- Augmentation types ---
augmentations = {
    "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
    "Vertical Flip": transforms.RandomVerticalFlip(p=1.0),
    "Rotation": transforms.RandomRotation(30),
    "Color Jitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    "Random Crop": transforms.RandomResizedCrop(size=(32,32), scale=(0.8,1.0))
}
 
# --- Apply augmentations ---
aug_images = {}
for name in augmentations:
    aug = augmentations[name]
    img = images[0]
    aug_img = aug(img)
    aug_images[name] = aug_img  # store augmented image

# --- Visualization ---
fig, axes = plt.subplots(1, 6, figsize=(15,5))

# Original image
axes[0].imshow(images[0].permute(1,2,0))
axes[0].axis('off')
axes[0].set_title("Original")

# Augmented images
i = 1
for name in aug_images:
    img = aug_images[name]
    axes[i].imshow(img.permute(1,2,0))
    axes[i].axis('off')
    axes[i].set_title(name)
    i = i + 1

plt.suptitle("Data Augmentation Types")
plt.show()
