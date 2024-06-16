class TrainLoop:
    def __init__(self, model, data_loader, epochs, val_data=None):
        self.model = model
        self.data_loader = data_loader
        self.epochs = epochs
        self.val_data = val_data

    def run(self):
        self.model.train(self.data_loader, self.epochs, self.val_data)
