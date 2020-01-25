"""
| Transformer Re-implementation code by HyeongGyu Froilan Choi

| Default Device : cuda
| As the model size is big, this model can be trained only on GPU
"""

"""
THINGS TO TALK ABOUT ON 1.28
1. Data Preprocessing : sorting by length & dropping
2. <sos><eos>the phenomenon
3. Label Smoothing
4. Save & Load Models
5. Dynamic Learning Rate --> Did not turn out nice
6. Miscellaneous : plot losses & print out examples
7. Q: further NLP preprocessing needed?  cf. punctuations : blank space in front of question/exclamation marks in French
"""

from config import get_args
from dataset import Dataset
from model import Transformer
from trainer import Trainer
import torch

def get_model(config, dataset):
    if config.reset_model :
        model = Transformer(config, dataset).cuda()
    else :
        model = torch.load('./out/{}.mdl'.format(config.model_name)).cuda()
    return model

def main():
    config = get_args()
    dataset = Dataset(config)
    model = get_model(config, dataset)
    trainer = Trainer(config, dataset, model)
    trainer.train()

if __name__ == '__main__' :
    main()