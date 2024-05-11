import numpy as np
import random

class Neuron:
    def __init__(self, bias=0.0):
        self.value = 0.0
        self.bias = bias
        self.incoming_links = []
        self.bias_gradient = 0.0

    def calculate_output(self):
        total_input = sum(link.weight * link.start_neuron.value for link in self.incoming_links)
        self.value = Neuron.sigmoid(total_input + self.bias)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

class Link:
    def __init__(self, start_neuron, end_neuron, weight=0.0):
        self.start_neuron = start_neuron
        self.end_neuron = end_neuron
        self.weight = weight
        self.weight_gradient = 0.0

class Layer:
    def __init__(self, num_neurons):
        self.neurons = [Neuron() for _ in range(num_neurons)]

    def connect_to(self, next_layer):
        for neuron in self.neurons:
            for next_neuron in next_layer.neurons:
                weight = random.uniform(0, 1)
                link = Link(neuron, next_neuron, weight)
                next_neuron.incoming_links.append(link)

class NeuralNetwork:
    def __init__(self, layer_sizes, initialize_random=True, weights_biases_file=None):
        self.layers = [Layer(size) for size in layer_sizes]

        for i in range(len(self.layers) - 1):
            self.layers[i].connect_to(self.layers[i + 1])

        if not initialize_random and weights_biases_file:
            self.load_weights_biases(weights_biases_file)

    def load_weights_biases(self, filename):
        with open(filename, 'r') as f:
            for layer in self.layers[1:]:
                for neuron in layer.neurons:
                    neuron.bias = float(f.readline().strip())
                    for link in neuron.incoming_links:
                        link.weight = float(f.readline().strip())

    def feedforward(self, input_values):
        for i, value in enumerate(input_values):
            self.layers[0].neurons[i].value = value

        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.calculate_output()

        return [neuron.value for neuron in self.layers[-1].neurons]

    def backpropagate(self, target_values, learning_rate):
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = target_values[i] - neuron.value
            neuron.bias_gradient = error * Neuron.sigmoid_derivative(neuron.value)
            for link in neuron.incoming_links:
                link.weight_gradient = neuron.bias_gradient * link.start_neuron.value

        for layer in reversed(self.layers[:-1]):
            for neuron in layer.neurons:
                downstream_gradient = sum(link.weight * link.end_neuron.bias_gradient for link in neuron.incoming_links)
                neuron.bias_gradient = downstream_gradient * Neuron.sigmoid_derivative(neuron.value)
                for link in neuron.incoming_links:
                    link.weight_gradient = neuron.bias_gradient * link.start_neuron.value

        for layer in self.layers[1:]:
            for neuron in layer.neurons:
                neuron.bias += learning_rate * neuron.bias_gradient
                for link in neuron.incoming_links:
                    link.weight += learning_rate * link.weight_gradient

    def train(self, input_values, target_values, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.feedforward(input_values)
            self.backpropagate(target_values, learning_rate)
            print(f"Epoch {epoch + 1}: {output}")

    def print_network(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}:")
            for j, neuron in enumerate(layer.neurons):
                incoming = [(link.start_neuron.value, link.weight) for link in neuron.incoming_links]
                print(f"  Neuron {j}: value={neuron.value}, bias={neuron.bias}, incoming={incoming}")

network = NeuralNetwork([2, 3, 1])
network.train([0.5, 0.1], [0.8], learning_rate=0.1, epochs=10)
network.print_network()
