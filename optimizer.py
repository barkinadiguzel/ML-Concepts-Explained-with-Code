import torch
import torch.nn as nn
import torch.optim as optim

# Simple linear model: y = wx + b
model = nn.Linear(1, 1)

# Loss function (MSE)
loss_fn = nn.MSELoss()

# Optimizer (Stochastic Gradient Descent - SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # y = 2x (true function)

# Training step (single epoch example)
for epoch in range(5):
    # Forward pass (make predictions)
    y_pred = model(x)

    # Calculate loss
    loss = loss_fn(y_pred, y)

    # Backward pass (zero gradients + compute gradients)
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
