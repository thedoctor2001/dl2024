import random
import math

class Layer:
    def __init__(self, size, next_layer_size=None, activation='tanh'):
        self.size = size
        self.activation = activation
        self.weights = []
        self.biases = []

        if next_layer_size:
            self.weights = [[0.0 for _ in range(size)] for _ in range(next_layer_size)]
            self.biases = [0.0 for _ in range(next_layer_size)]

    def initialize_random(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = random.uniform(0, 1)
            self.biases[i] = random.uniform(0, 1)

    def initialize_from_file(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def activate(self, x):
        if self.activation == 'tanh':
            return math.tanh(x)

        return x

    def feedforward(self, inputs):
        outputs = []
        for i in range(len(self.weights)):
            activation_sum = sum(w * inp for w, inp in zip(self.weights[i], inputs)) + self.biases[i]
            outputs.append(self.activate(activation_sum))
        return outputs

class NeuralNetwork:
    def __init__(self, structure):
        self.layers = []
        for i in range(len(structure)-1):
            self.layers.append(Layer(structure[i], structure[i+1]))

    def initialize_weights(self, method='random', filename=None):
        if method == 'from_file':
            with open(filename, 'r') as f:
                for layer in self.layers:
                    weights = []
                    for _ in range(len(layer.weights)):
                        weights.append(list(map(float, f.readline().strip().split())))
                    biases = list(map(float, f.readline().strip().split()))
                    layer.initialize_from_file(weights, biases)
        elif method == 'random':
            for layer in self.layers:
                layer.initialize_random()

    def feedforward(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.feedforward(activations)
        return activations

structure = [3, 5, 2]
nn = NeuralNetwork(structure)
nn.initialize_weights('random')
inputs = [0.5, -0.3, 0.1]
output = nn.feedforward(inputs)
print(output)