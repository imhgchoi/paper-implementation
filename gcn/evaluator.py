from sklearn.metrics import f1_score
import numpy as np

class Evaluator():
    def __init__(self, config):
        self.config = config

    def get_accuracy(self, pred, y):
        accuracy = lambda x : np.sum(x)/len(x)
        match = np.argmax(pred.cpu().detach().numpy(), axis=1) == y.numpy()
        return accuracy(match)

    def get_fscore(self, pred, y):
        return f1_score(y.numpy(), np.argmax(pred.cpu().detach().numpy(), axis=1), average='macro')

    def test(self, model, loader):
        model.eval()
        for i, batch in enumerate(loader) :
            out = model(batch)
            accuracy = self.get_accuracy(out[batch.test_mask], batch.y[batch.test_mask])
            fscore = self.get_fscore(out[batch.test_mask], batch.y[batch.test_mask])
            print("\tTEST ACCURACY = {0:.4f} | F1 = {1:.2f}".format(accuracy, fscore))