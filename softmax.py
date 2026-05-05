import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1000, 1001, 1002])
print(softmax(x))

#overflow occurs in np.exp(x) when x is large, resulting in inf values. This leads to NaN values in the output of softmax. To prevent this, we can subtract the maximum value from x before applying the exponential function:

import torch
x = torch.tensor([1000.0, 1001.0, 1002.0])
print(torch.softmax(x, dim=0))
#tensor([0.0900, 0.2447, 0.6652])
