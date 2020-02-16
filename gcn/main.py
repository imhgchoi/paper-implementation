"""
| Graph Convolutional Network Re-implementation code by HyeongGyu Froilan Choi

"""
from config import get_args
from dataset import Dataset
from models.pygGCN import GCNpyg
from models.rawGCN import GCNraw
from evaluator import Evaluator
from trainer import Trainer
import torch
import warnings
warnings.filterwarnings('ignore')

def get_model(config, dataset, device):
    if config.use_pyg :
        model = GCNpyg(config, dataset, device)
    else :
        model = GCNraw(config, dataset, device)
    if not config.reset_model :
        model.load_state_dict(torch.load('./out/{}.mdl'.format(config.model_name)))
        model.eval()

    return model


def main():
    config = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dataset(config)
    model = get_model(config, dataset, device)
    evaluator = Evaluator(config)
    trainer = Trainer(config, dataset.loader, evaluator)
    trained_model = trainer.train(model)
    torch.save(trained_model.state_dict(), './out/models/{}_{}.mdl'.format(config.model_name, config.datatype))


if __name__ == "__main__" :
    main()
