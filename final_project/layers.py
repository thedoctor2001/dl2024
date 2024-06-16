import numpy as np
from utils import Initializer


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError


class Conv2D(Layer):
    def __init__(
        self, num_filters, filter_size, stride=1, padding=0, initialization="xavier"
    ):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        if initialization == "xavier":
            self.filters = Initializer.xavier(
                (num_filters, filter_size, filter_size, 1)
            )
        elif initialization == "he":
            self.filters = Initializer.he((num_filters, filter_size, filter_size, 1))
        self.x = None
        self.input_padded = None

    def forward(self, x):
        self.x = x
        self.input_padded = np.pad(
            x,
            (
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
                (0, 0),
            ),
        )
        output_shape = (
            (x.shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1,
            (x.shape[2] - self.filter_size + 2 * self.padding) // self.stride + 1,
            self.num_filters,
        )
        output = np.zeros((x.shape[0], *output_shape))
        for i in range(
            0, self.input_padded.shape[1] - self.filter_size + 1, self.stride
        ):
            for j in range(
                0, self.input_padded.shape[2] - self.filter_size + 1, self.stride
            ):
                region = self.input_padded[
                    :, i : i + self.filter_size, j : j + self.filter_size
                ]
                for k in range(self.num_filters):
                    output[:, i // self.stride, j // self.stride, k] = np.sum(
                        region * self.filters[k], axis=(1, 2, 3)
                    )
        return output

    def backward(self, output_grad):
        grad_input = np.zeros_like(self.input_padded)
        grad_filters = np.zeros_like(self.filters)
        for i in range(
            0, self.input_padded.shape[1] - self.filter_size + 1, self.stride
        ):
            for j in range(
                0, self.input_padded.shape[2] - self.filter_size + 1, self.stride
            ):
                region = self.input_padded[
                    :, i : i + self.filter_size, j : j + self.filter_size
                ]
                for k in range(self.num_filters):
                    grad_filters[k] += np.sum(
                        region
                        * (output_grad[:, i // self.stride, j // self.stride, k])[
                            :, None, None, None
                        ],
                        axis=0,
                    )
                    grad_input[
                        :, i : i + self.filter_size, j : j + self.filter_size
                    ] += (
                        output_grad[:, i // self.stride, j // self.stride, k][
                            :, None, None, None
                        ]
                        * self.filters[k]
                    )
        if self.padding != 0:
            grad_input = grad_input[
                :, self.padding : -self.padding, self.padding : -self.padding, :
            ]
        return grad_input, grad_filters


class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        self.x = x
        output_shape = (
            x.shape[0],
            (x.shape[1] - self.pool_size) // self.stride + 1,
            (x.shape[2] - self.pool_size) // self.stride + 1,
            x.shape[3],
        )
        output = np.zeros(output_shape)
        for i in range(0, x.shape[1] - self.pool_size + 1, self.stride):
            for j in range(0, self.x.shape[2] - self.pool_size + 1, self.stride):
                region = x[:, i : i + self.pool_size, j : j + self.pool_size]
                output[:, i // self.stride, j // self.stride] = np.max(
                    region, axis=(1, 2)
                )
        return output

    def backward(self, output_grad):
        grad_input = np.zeros_like(self.x)
        for i in range(0, self.x.shape[1] - self.pool_size + 1, self.stride):
            for j in range(0, self.x.shape[2] - self.pool_size + 1, self.stride):
                region = self.x[:, i : i + self.pool_size, j : j + self.pool_size]
                max_region = np.max(region, axis=(1, 2), keepdims=True)
                mask = region == max_region
                grad_input[:, i : i + self.pool_size, j : j + self.pool_size] += (
                    output_grad[:, i // self.stride, j // self.stride][:, None, None, :]
                    * mask
                )
        return grad_input, None


class Dense(Layer):
    def __init__(self, input_dim, output_dim, initialization="xavier"):
        if initialization == "xavier":
            self.weights = Initializer.xavier((input_dim, output_dim))
        elif initialization == "he":
            self.weights = Initializer.he((input_dim, output_dim))
        self.biases = np.zeros((1, output_dim))
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, output_grad):
        grad_input = np.dot(output_grad, self.weights.T)
        grad_weights = np.dot(self.x.T, output_grad)
        grad_biases = np.sum(output_grad, axis=0, keepdims=True)
        return grad_input, (grad_weights, grad_biases)


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, output_grad):
        return output_grad.reshape(self.input_shape), None
