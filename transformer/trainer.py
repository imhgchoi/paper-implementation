
class Trainer():
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model

    def train(self):
        for batch in self.dataset.TRAIN :
            en = batch.en  # sentence length * batch size
            fr = batch.fr  # sentence length * batch size

            self.model(fr)
            exit()