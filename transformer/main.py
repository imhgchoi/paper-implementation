"""
| Transformer Re-implementation code by HyeongGyu Froilan Choi

| Default Device : cuda
| As the model size is big, this model can be trained only on GPU
"""


from config import get_args
from dataset import Dataset
from model import Transformer
from trainer import Trainer
from evaluator import Evaluator
import torch
import warnings
warnings.filterwarnings('ignore')

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
    evaluator = Evaluator(config, dataset)
    trainer = Trainer(config, dataset, model, evaluator)
    trainer.train()

if __name__ == '__main__' :
    main()