from activations import ReLU, Softmax
from data_loader import DataLoader, Dataset, BatchLoader
from layers import Conv2D, MaxPool2D, Flatten, Dense
from models import Model
from train_loop import TrainLoop
from utils import Loss, SGD, Adam


def load_and_prepare_data():
    train_images_path = 'data/train-images-idx3-ubyte.gz'
    train_labels_path = 'data/train-labels-idx1-ubyte.gz'
    test_images_path = 'data/t10k-images-idx3-ubyte.gz'
    test_labels_path = 'data/t10k-labels-idx1-ubyte.gz'

    data_loader = DataLoader(train_images_path, train_labels_path)
    train_images, train_labels = data_loader.load_data()
    train_dataset = Dataset(train_images, train_labels)

    data_loader = DataLoader(test_images_path, test_labels_path)
    test_images, test_labels = data_loader.load_data()
    test_dataset = Dataset(test_images, test_labels)

    return train_dataset, test_dataset


def main():
    train_dataset, test_dataset = load_and_prepare_data()
    train_loader = BatchLoader(train_dataset, batch_size=32)
    val_data = (test_dataset.images, test_dataset.labels)

    model = Model()
    model.add(Conv2D(num_filters=8, filter_size=3, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Flatten())
    model.add(Dense(input_dim=14 * 14 * 8, output_dim=10))
    model.add(Softmax())

    optimizer = Adam(learning_rate=0.0001, beta1=0.99, beta2=0.9999, epsilon=1e-6, regularization_strength=0.0001)

    model.compile(loss=Loss, optimizer=optimizer)

    train_loop = TrainLoop(model, train_loader, epochs=10, val_data=val_data)
    train_loop.run()


if __name__ == '__main__':
    main()
