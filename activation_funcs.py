import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)

relu = F.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

plt.figure(figsize=(8,4))
plt.plot(x.numpy(), relu.numpy(), label="ReLU")
plt.plot(x.numpy(), sigmoid.numpy(), label="Sigmoid")
plt.plot(x.numpy(), tanh.numpy(), label="Tanh")
plt.title("Activation Functions")
plt.legend()
plt.grid(True)
plt.show()

