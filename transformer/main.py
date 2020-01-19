from config import get_args
from dataset import Dataset
from model import Transformer
from trainer import Trainer

"""
Default : cuda
As the model size is big, this model can be trained only on GPU
"""

def main():
    config = get_args()
    dataset = Dataset(config)
    model = Transformer(config, dataset).cuda()
    trainer = Trainer(config, dataset, model)
    trainer.train()

if __name__ == '__main__' :
    main()