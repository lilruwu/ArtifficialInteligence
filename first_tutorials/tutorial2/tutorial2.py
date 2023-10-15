# In the above examples, we had to manually implement both the forward and backward passes of our neural network. 
# Manually implementing the backward pass is not a big deal for a small two-layer network, 
# but can quickly get very hairy for large complex networks.

# Thankfully, we can use automatic differentiation to automate the computation of backward passes in neural networks. 
# The autograd package in PyTorch provides exactly this functionality. When using autograd, 
# the forward pass of your network will define a computational graph; nodes in the graph will be Tensors, 
# and edges will be functions that produce output Tensors from input Tensors. 
# Backpropagating through this graph then allows you to easily compute gradients.

# This sounds complicated, it’s pretty simple to use in practice. 
# Each Tensor represents a node in a computational graph. If x is a Tensor that has x.requires_grad=True 
# then x.grad is another Tensor holding the gradient of x with respect to some scalar value.

# Here we use PyTorch Tensors and autograd to implement our fitting sine wave 
# with third order polynomial example; now we no longer need to manually implement the backward pass through the network:

# -*- coding: utf-8 -*-
import torch
import math
import datetime

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), dtype=dtype, requires_grad=True)
b = torch.randn((), dtype=dtype, requires_grad=True)
c = torch.randn((), dtype=dtype, requires_grad=True)
d = torch.randn((), dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

# Save the result into a file
output_file = f"output.txt"

with open(output_file, 'a') as f:
    f.write(f'Result ({datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}): y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3\n')


