import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNraw(nn.Module):
    def __init__(self, config, dataset, device):
        super(GCNraw, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device

        conv_dim = [dataset.dataset.num_node_features] + self.config.conv_layers
        fc_dim = [self.config.conv_layers[-1]] + self.config.fc_layers + [dataset.dataset.num_classes]

        self.conv = torch.nn.Sequential()
        for i in range(len(conv_dim) - 1):
            self.add_conv(i, conv_dim[i], conv_dim[i + 1])

        self.fc = torch.nn.Sequential()
        for i in range(len(fc_dim) - 1):
            self.add_fc(i, fc_dim[i], fc_dim[i + 1])

        self.print_model()
        self.to(device)

    def add_conv(self, i, d_in, d_out):
        if self.config.use_base :
            self.conv.add_module("conv{}".format(i), baseGraphConv(d_in, d_out))
        else :
            self.conv.add_module("conv{}".format(i), rawGraphConv(d_in, d_out, self.device))
        self.conv.add_module("relu{}".format(i), nn.ReLU())
        self.conv.add_module("bn{}".format(i), nn.BatchNorm1d(d_out))
        self.conv.add_module("dropout{}".format(i), nn.Dropout(self.config.dropout))

    def add_fc(self, i, d_in, d_out):
        self.fc.add_module("fc{}".format(i), nn.Linear(d_in, d_out))
        self.fc.add_module("relu{}".format(i), nn.ReLU())
        self.fc.add_module("dropout{}".format(i), nn.Dropout(self.config.dropout))

    def print_model(self):
        print(":::: MODEL ARCHITECTURE ::::")
        for layer in self.children() :
            print(layer)
        print("\n\n")

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)

        for layer in self.conv :
            if 'GraphConv' in str(layer) :
                x = layer(x, edge_index)
            else :
                x = layer(x)

        for layer in self.fc :
            x = layer(x)
        return x

class rawGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(rawGraphConv, self).__init__()
        self.device = device

        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        # get Adjacency matrix
        A = torch.eye(x.shape[0]).to(self.device)
        for i, j in torch.t(edge_index):
            A[i.item(), j.item()] = 1

        # get Degree matrix
        D = torch.sqrt(torch.diag(torch.sum(A, axis=1)))

        out = self.fc(torch.mm(D * A * D, x))

        return out

# reference : https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
class baseGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(baseGraphConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out