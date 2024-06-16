import math

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        grads = []
        for layer in reversed(self.layers):
            loss_grad, layer_grads = layer.backward(loss_grad)
            if layer_grads is not None:
                grads.extend(layer_grads)
        return grads

    def compute_loss(self, predictions, labels):
        loss = self.loss.cross_entropy(predictions, labels)
        l2_loss = self.compute_l2_loss()
        return loss + l2_loss

    def compute_l2_loss(self):
        l2_loss = 0
        for layer in self.layers:
            if hasattr(layer, "weights"):
                l2_loss += self.optimizer.regularization_strength * sum(
                    weight ** 2 for row in layer.weights for weight in row
                )
        return l2_loss

    def train(self, data_loader, epochs, val_data=None):
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0
            total_samples = 0
            batch_count = 0

            for images, labels in data_loader:
                predictions = self.forward(images)
                loss = self.compute_loss(predictions, labels)
                epoch_loss += loss
                loss_grad = self.loss.cross_entropy_grad(predictions, labels)
                grads = self.backward(loss_grad)
                self.optimizer.update(self.get_params(), grads)

                correct_predictions += sum(
                    1 for pred, label in zip(predictions, labels) if pred.index(max(pred)) == label
                )
                total_samples += len(labels)
                batch_count += 1

            average_loss = epoch_loss / batch_count
            accuracy = correct_predictions / total_samples

            if val_data:
                val_loss, val_accuracy = self.evaluate(val_data)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

            data_loader.on_epoch_end()

    def evaluate(self, val_data):
        val_images, val_labels = val_data
        predictions = self.forward(val_images)
        loss = self.compute_loss(predictions, val_labels)
        accuracy = self.calculate_accuracy(predictions, val_labels)
        return loss, accuracy

    @staticmethod
    def calculate_accuracy(predictions, labels):
        prediction_labels = [pred.index(max(pred)) for pred in predictions]
        return sum(1 for pred, label in zip(prediction_labels, labels) if pred == label) / len(labels)

    def get_params(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                params.append(layer.weights)
            if hasattr(layer, "biases"):
                params.append(layer.biases)
        return params