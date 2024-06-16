import numpy as np


class Initializer:
    @staticmethod
    def xavier(shape):
        return np.random.randn(*shape) * np.sqrt(2 / np.sum(shape))

    @staticmethod
    def he(shape):
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])


class Loss:
    @staticmethod
    def cross_entropy(predictions, labels):
        num_samples = predictions.shape[0]
        clipped_predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        correct_confidences = clipped_predictions[range(num_samples), labels]
        return -np.mean(np.log(correct_confidences))

    @staticmethod
    def cross_entropy_grad(predictions, labels):
        num_samples = predictions.shape[0]
        grad = predictions.copy()
        grad[range(num_samples), labels] -= 1
        return grad / num_samples


class Optimizer:
    def __init__(self, learning_rate=0.01, regularization_strength=0.0):
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength

    def update(self, params, grads):
        raise NotImplementedError("This method should be overridden by subclasses")


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, decay=0.0, regularization_strength=0.0):
        super().__init__(learning_rate, regularization_strength)
        self.initial_learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay
        self.velocities = None
        self.iterations = 0

    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param in params]

        self.iterations += 1
        if self.decay:
            self.learning_rate = self.initial_learning_rate / (1 + self.decay * self.iterations)

        for param, grad, velocity in zip(params, grads, self.velocities):
            if self.regularization_strength:
                grad += self.regularization_strength * param
            if self.momentum:
                velocity = self.momentum * velocity + grad
                update_value = velocity
                if self.nesterov:
                    update_value = self.momentum * velocity + grad
                param -= self.learning_rate * update_value
            else:
                param -= self.learning_rate * grad
            velocity[:] = velocity


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, regularization_strength=0.0):
        super().__init__(learning_rate, regularization_strength)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            if self.regularization_strength:
                grad += self.regularization_strength * param

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
