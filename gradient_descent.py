# mini_gradient_descent.py
import torch
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Function and starting parameter
# -----------------------------
# f(x) = x^2 (minimum at x=0)
x = torch.tensor([5.0], requires_grad=True)  # starting point
learning_rate = 0.1
steps = 20

# -----------------------------
# 2️⃣ Gradient Descent steps
# -----------------------------
x_history = []

for i in range(steps):
    y = x**2               # function
    y.backward()           # compute gradient
    x_history.append(x.item())
    
    # gradient descent update
    with torch.no_grad():
        x -= learning_rate * x.grad
    x.grad.zero_()         # reset gradient

# -----------------------------
# 3️⃣ Visualization
# -----------------------------
xs = torch.linspace(-6, 6, 100)
ys = xs**2

plt.plot(xs, ys, label='y = x^2')
plt.scatter(x_history, [x**2 for x in x_history], color='red', label='Gradient Descent Steps')
plt.title('Mini Gradient Descent Visualization')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
