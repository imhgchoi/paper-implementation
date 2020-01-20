import torch
import torch.nn as nn
import torch.optim as optim

class Trainer():
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model
        print('\n\n')

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epoch_num) :
            for idx, batch in enumerate(self.dataset.TRAIN) :
                fr = batch.fr  # sentence length * batch size
                en = batch.en  # sentence length * batch size

                optimizer.zero_grad()

                out = self.model(fr, en)  # batch size * sentence length * vocab size
                out = out.transpose(1,2)  # batch size * vocab size * sentence length
                pred = torch.argmax(out, dim=2).cpu()

                en = en.transpose(0,1)  # batch size * sentence length
                loss = criterion(out, en)
                loss.backward()

                optimizer.step()

                print('EPOCH {0} [ batch {1} ] : LOSS = {2:.5f}'.format(epoch, idx, loss.item()))

