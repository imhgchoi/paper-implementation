from torch_geometric import datasets
from torch_geometric.data import DataLoader
import pickle
import shutil

class Dataset():
    def __init__(self, config):
        self.config = config
        self.dataset = self.get_dataset(config.datatype)
        self.loader = self.get_loader()

    def get_dataset(self, datatype):
        if self.config.reset_data :
            shutil.rmtree(self.config.datadir+'processed/')
            if datatype == 'cora' :
                data = datasets.Planetoid(root=self.config.datadir, name='Cora')
            elif datatype == 'citeseer':
                data = datasets.Planetoid(root=self.config.datadir, name='CiteSeer')
                print(data)
            elif datatype == 'pubmed' :
                data = datasets.Planetoid(root=self.config.datadir, name='PubMed')
            else :
                raise ValueError

            with open(self.config.datadir+'dat.pkl', 'wb') as f :
                pickle.dump(data, f)
            f.close()
        else :
            with open(self.config.datadir+'dat.pkl', 'rb') as f :
                data = pickle.load(f)

        return data


    def get_loader(self):
        return DataLoader(self.dataset, batch_size=self.config.batchsize, shuffle=True)
