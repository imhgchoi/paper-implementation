import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool

class GCNpyg(nn.Module):
    def __init__(self, config, dataset, device):
        super(GCNpyg, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        conv_dim = [dataset.dataset.num_node_features] + self.config.conv_layers
        fc_dim = [self.config.conv_layers[-1]] + self.config.fc_layers + [dataset.dataset.num_classes]

        self.conv = torch.nn.Sequential()
        for i in range(len(conv_dim)-1) :
            self.add_conv(i, conv_dim[i], conv_dim[i+1])

        self.fc = torch.nn.Sequential()
        for i in range(len(fc_dim)-1) :
            self.add_fc(i, fc_dim[i], fc_dim[i+1])

        self.to(device)

    def add_conv(self, i, d_in, d_out):
        self.conv.add_module("conv{}".format(i), GCNConv(d_in, d_out, cached=False))
        self.conv.add_module("relu{}".format(i), nn.ReLU())
        self.conv.add_module("bn{}".format(i), nn.BatchNorm1d(d_out))
        self.conv.add_module("dropout{}".format(i), nn.Dropout(self.config.dropout))

    def add_fc(self, i, d_in, d_out):
        self.fc.add_module("fc{}".format(i), nn.Linear(d_in, d_out))
        self.fc.add_module("relu{}".format(i), nn.ReLU())
        self.fc.add_module("dropout{}".format(i), nn.Dropout(self.config.dropout))

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        for layer in self.conv :
            if 'GCNConv' in str(layer) :
                x = layer(x, edge_index)
            else :
                x = layer(x)

        for layer in self.fc :
            x = layer(x)
        return x