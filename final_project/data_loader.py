import gzip
import numpy as np


class DataLoader:
    def __init__(self, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path

    def load_data(self):
        with gzip.open(self.images_path, 'rb') as img_path:
            images = np.frombuffer(img_path.read(), np.uint8, offset=16)
            images = images.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

        with gzip.open(self.labels_path, 'rb') as lbl_path:
            labels = np.frombuffer(lbl_path.read(), np.uint8, offset=8)

        return images, labels


class Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class BatchLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            end = min(start + self.batch_size, len(self.indices))
            batch_indices = self.indices[start:end]
            yield self.dataset.images[batch_indices], self.dataset.labels[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
