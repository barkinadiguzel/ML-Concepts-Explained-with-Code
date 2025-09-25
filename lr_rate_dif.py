import torch
import matplotlib.pyplot as plt
import numpy as np

# Basit loss ve gradient
def loss(x):
    return x ** 2

def grad(x):
    return 2 * x

# Başlangıç weight
x_init = 5.0
epochs = 20

# --- Learning rate 0.1 ---
x1 = torch.tensor([x_init], dtype=torch.float32)
x1_list = []
y1_list = []

for epoch in range(epochs):
    x1 = x1 - 0.1 * grad(x1)
    x1_list.append(x1.item())
    y1_list.append(loss(x1).item())

# --- Learning rate 0.01 ---
x2 = torch.tensor([x_init], dtype=torch.float32)
x2_list = []
y2_list = []

for epoch in range(epochs):
    x2 = x2 - 0.01 * grad(x2)
    x2_list.append(x2.item())
    y2_list.append(loss(x2).item())

# Loss yüzeyi için x aralığı
x_vals = np.linspace(-6,6,100)
y_vals = x_vals**2
plt.plot(x_vals, y_vals, color='gray', linestyle='--', label="Loss Surface")

# Noktaları plotla
plt.scatter(x1_list, y1_list, color='red', s=50, label="lr=0.1")
plt.scatter(x2_list, y2_list, color='green', s=50, label="lr=0.01")

plt.xlabel("Weight (x)")
plt.ylabel("Loss (y = x^2)")
plt.title("Effect of Learning Rate on Gradient Descent")
plt.legend()
plt.show()
