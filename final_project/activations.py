import numpy as np


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, output_grad):
        return output_grad * (self.x > 0), None


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    def backward(self, output_grad):
        num_samples = self.output.shape[0]
        grad = np.zeros_like(output_grad)
        for i in range(num_samples):
            y = self.output[i]
            jacobian_matrix = np.diag(y) - np.outer(y, y)
            grad[i] = np.dot(jacobian_matrix, output_grad[i])
        return grad, None
