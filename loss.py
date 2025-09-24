import torch
import torch.nn as nn

# Ground truth values (y)
y_true = torch.tensor([2.5, 0.0, 2.1, 7.8])

# Predicted values from the model (y_hat)
y_pred = torch.tensor([2.4, 0.1, 2.0, 7.9])

# Define Mean Squared Error (MSE) Loss
loss_fn = nn.MSELoss()

# Calculate the loss between predictions and ground truth
loss = loss_fn(y_pred, y_true)

print("MSE Loss:", loss.item())
