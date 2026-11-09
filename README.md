# XOR activation test

A two-layer neural network written from scratch in NumPy, trained on XOR twice: once with no activation function between the layers, and once with ReLU.

Without an activation, two weight matrices multiply into one (a 2x8 matrix times an 8x1 matrix is just a 2x1 matrix), so the network collapses into a single linear layer and can't learn XOR.

## Results

| Version | Accuracy on XOR |
|---|---|
| No activation | 50% (25% on some seeds) |
| ReLU | 100% on every seed tested |

## Run

```
pip3 install numpy
python3 xor_net.py
```
