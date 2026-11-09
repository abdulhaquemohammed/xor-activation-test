"""
Does a neural network need an activation function?
Tests the claim: without one, stacking layers collapses into a single layer,
so the network can't learn XOR.
"""
import numpy as np

# XOR: output is 1 only when exactly one input is 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train(use_relu, hidden=8, lr=0.5, epochs=5000, seed=0):
    rng = np.random.default_rng(seed)
    # Layer 1: 2 inputs -> hidden units. Layer 2: hidden units -> 1 output
    W1 = rng.normal(0, 1, (2, hidden)); b1 = np.zeros((1, hidden))
    W2 = rng.normal(0, 1, (hidden, 1)); b2 = np.zeros((1, 1))

    for _ in range(epochs):
        # Forward pass: each layer is a matrix multiplication
        z1 = X @ W1 + b1
        h = np.maximum(0, z1) if use_relu else z1   # ReLU, or nothing
        out = sigmoid(h @ W2 + b2)

        # Backward pass (backpropagation): how much each weight caused the error
        d_out = (out - y) / len(X)
        dW2 = h.T @ d_out; db2 = d_out.sum(0, keepdims=True)
        d_h = d_out @ W2.T
        if use_relu:
            d_h = d_h * (z1 > 0)
        dW1 = X.T @ d_h; db1 = d_h.sum(0, keepdims=True)

        # Gradient descent: nudge every weight to reduce the error
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    preds = (out > 0.5).astype(int)
    acc = (preds == y).mean() * 100
    return out.ravel().round(2), acc, W1, W2


for name, relu in [("No activation", False), ("With ReLU", True)]:
    out, acc, W1, W2 = train(relu)
    print(f"{name}: outputs {out} -> accuracy {acc:.0f}%")

# The collapse: two weight matrices with nothing between them are just one matrix
_, _, W1, W2 = train(False)
print("\nW1 is", W1.shape, "and W2 is", W2.shape, "-> W1 @ W2 is", (W1 @ W2).shape,
      "= a single layer straight from 2 inputs to 1 output")
