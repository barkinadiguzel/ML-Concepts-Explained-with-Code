"""
Early Stopping
==============
Stops training when validation loss stops improving to prevent overfitting.

How it works:
- Monitor validation loss during training
- If val loss doesn't improve for 'patience' epochs, stop training
- Restore the best model weights from before overfitting started

Benefits:
- Prevents overfitting automatically
- Saves training time
- No need to manually pick the right number of epochs
"""
 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Generate synthetic data
X_train = torch.randn(100, 10)
y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)
X_val = torch.randn(30, 10)
y_val = (X_val.sum(dim=1) > 0).float().unsqueeze(1)

# Simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Training with early stopping
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience = 5
patience_counter = 0
best_epoch = 0

for epoch in range(100):
    # Train
    model.train()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Validate
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_epoch = epoch
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        print(f"Best epoch was {best_epoch} with val loss {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)
        break

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', alpha=0.7)
plt.plot(val_losses, label='Val Loss', alpha=0.7)
plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping: Training stopped when validation loss stopped improving')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('early_stopping.png', dpi=150)
plt.show()

print(f"\nTraining stopped after {len(train_losses)} epochs")
print(f"Without early stopping, would have trained for 100 epochs")
