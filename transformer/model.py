import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Transformer(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.dataset = dataset

        self.input_embedding = nn.Embedding(self.dataset.fr_vocabsize, self.config.emb_dim)
        self.encoders = nn.ModuleList([Encoder(config) for _ in range(self.config.N)])
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(self.config.N)])


    def pos_enc(self, em):
        pe = torch.zeros(em.shape[0],em.shape[2])
        for i in range(pe.shape[0]) :
            for j in range(pe.shape[1]) :
                if j % 2 == 0 :
                    pe[i, j] = np.sin(i / (10000 ** (j/self.config.emb_dim)))
                else :
                    pe[i, j] = np.cos(i / (10000 ** ((j-1)/self.config.emb_dim)))
        return em + pe.unsqueeze(1).repeat(1,em.shape[1],1).cuda()

    def forward(self, x):
        input_emb = self.input_embedding(x)  # sent length * batch size * embedding dim
        enc_in = self.pos_enc(input_emb)     # sent length * batch size * embedding dim

        for enc in self.encoders :
            enc_in = enc(enc_in)
        enc_out = enc_in


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def multi_head_attention(self, Q, K, V):
        pass

    def forward(self, x):
        Q, K, V = x, x, x
        
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        pass