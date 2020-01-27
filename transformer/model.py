import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Transformer(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.config = config
        self.fr_vocab_size = dataset.fr_vocabsize
        self.en_vocab_size = dataset.en_vocabsize

        self.input_embedding = nn.Embedding(self.fr_vocab_size, self.config.emb_dim)
        self.output_embedding = nn.Embedding(self.en_vocab_size, self.config.emb_dim)
        self.encoders = nn.ModuleList([Encoder(config) for _ in range(self.config.N)])
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(self.config.N)])

        self.linear = nn.Linear(self.config.emb_dim, self.en_vocab_size)

    def pos_enc(self, em):
        pe = torch.zeros(em.shape[0],em.shape[2])
        for i in range(pe.shape[0]) :
            for j in range(pe.shape[1]) :
                if j % 2 == 0 :
                    pe[i, j] = np.sin(i / (10000 ** (j/self.config.emb_dim)))
                else :
                    pe[i, j] = np.cos(i / (10000 ** ((j-1)/self.config.emb_dim)))
        return em + pe.unsqueeze(1).repeat(1,em.shape[1],1).cuda()

    def forward(self, fr, en):
        input = fr[1:]

        # ENCODER
        input_emb = self.input_embedding(input)  # sent length * batch size * embedding dim
        enc_in = self.pos_enc(input_emb)     # sent length * batch size * embedding dim
        enc_in = enc_in.transpose(0,1)       # batch size * sent length * embedding dim

        for enc in self.encoders :
            enc_in = enc(enc_in)
        enc_out = enc_in        # batch size * sent length * embedding dim

        # DECODER
        output_emb = self.output_embedding(en)  # sent length * batch size * embedding dim
        dec_in = self.pos_enc(output_emb)     # sent length * batch size * embedding dim
        dec_in = dec_in.transpose(0,1)       # batch size * sent length * embedding dim

        for dec in self.decoders :
            dec_in = dec(dec_in, enc_out)
        dec_out = dec_in
        return self.linear(dec_out)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wqs = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                  for _ in range(self.config.headnum)])
        self.wks = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                  for _ in range(self.config.headnum)])
        self.wvs = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                  for _ in range(self.config.headnum)])
        self.softmax = nn.ModuleList([nn.Softmax(dim=2) for _ in range(self.config.headnum)])

        self.enc_dropout1 = nn.Dropout(self.config.enc_dropout)
        self.enc_norm1 = nn.LayerNorm(self.config.emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.config.emb_dim, self.config.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.config.emb_dim, self.config.emb_dim, bias=True)
        )

        self.enc_dropout2 = nn.Dropout(self.config.enc_dropout)
        self.enc_norm2 = nn.LayerNorm(self.config.emb_dim)

    def multi_head_attention(self, Q, K, V):
        Qs = [wq(Q) for wq in self.wqs]  # headnum * batchsize * sent length * (embedding dim / headnum)
        Ks = [wk(K) for wk in self.wks]  # headnum * batchsize * sent length * (embedding dim / headnum)
        Vs = [wv(V) for wv in self.wvs]  # headnum * batchsize * sent length * (embedding dim / headnum)

        mm_scaled = [torch.matmul(Q, K.transpose(1, 2))/(self.config.emb_dim/self.config.headnum)**(1/2)
                     for Q, K in zip(Qs, Ks)]          # headnum * batchsize * sent length * sent length
        softmaxed = [sm(x) for x, sm in zip(mm_scaled, self.softmax)] # headnum * batchsize * sent length * sent length
        attention = [torch.matmul(sm, v) for sm, v in zip(softmaxed, Vs)] # headnum * batchsize * sent length * (embedding dim / headnum)
        return torch.cat(attention, dim=2) # batchsize * sent length * embedding dim

    def add_norm1(self, attention, x):
        do = self.enc_dropout1(attention) # batchsize * sent length * embedding dim
        x_do = x + do   # batchsize * sent length * embedding dim
        return self.enc_norm1(x_do)

    def add_norm2(self, fc1, x):
        do = self.enc_dropout2(fc1) # batchsize * sent length * embedding dim
        x_do = x + do   # batchsize * sent length * embedding dim
        return self.enc_norm2(x_do)

    def forward(self, x):
        Q, K, V = x, x, x   # batch size * sent length * embedding dim
        attention = self.multi_head_attention(Q, K, V)  # batchsize * sent length * embedding dim
        add_norm1 = self.add_norm1(attention, x)  # batchsize * sent length * embedding dim
        fc1 = self.fc(add_norm1) # batchsize * sent length * embedding dim
        out = self.add_norm2(fc1, add_norm1) # batchsize * sent length * embedding dim
        return out


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.masked_wqs = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                         for _ in range(self.config.headnum)])
        self.masked_wks = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                         for _ in range(self.config.headnum)])
        self.masked_wvs = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                         for _ in range(self.config.headnum)])
        self.softmax1 = nn.ModuleList([nn.Softmax(dim=2) for _ in range(self.config.headnum)])
        self.dec_dropout1 = nn.Dropout(self.config.dec_dropout)
        self.dec_norm1 = nn.LayerNorm(self.config.emb_dim)

        self.wqs = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                  for _ in range(self.config.headnum)])
        self.wks = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                  for _ in range(self.config.headnum)])
        self.wvs = nn.ModuleList([nn.Linear(self.config.emb_dim, int(self.config.emb_dim/self.config.headnum))
                                  for _ in range(self.config.headnum)])
        self.softmax2 = nn.ModuleList([nn.Softmax(dim=2) for _ in range(self.config.headnum)])
        self.dec_dropout2 = nn.Dropout(self.config.dec_dropout)
        self.dec_norm2 = nn.LayerNorm(self.config.emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(self.config.emb_dim, self.config.emb_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.config.emb_dim, self.config.emb_dim, bias=True)
        )

        self.dec_dropout3 = nn.Dropout(self.config.dec_dropout)
        self.dec_norm3 = nn.LayerNorm(self.config.emb_dim)

    def masked_multi_head_attention(self, Q, K, V):
        Qs = [wq(Q) for wq in self.masked_wqs]  # headnum * batchsize * sent length * (embedding dim / headnum)
        Ks = [wk(K) for wk in self.masked_wks]  # headnum * batchsize * sent length * (embedding dim / headnum)
        Vs = [wv(V) for wv in self.masked_wvs]  # headnum * batchsize * sent length * (embedding dim / headnum)

        mm_scaled = [torch.matmul(Q, K.transpose(1, 2))/(self.config.emb_dim/self.config.headnum)**(1/2)
                     for Q, K in zip(Qs, Ks)]          # headnum * batchsize * sent length * sent length
        # masking
        mask = torch.zeros(mm_scaled[0].shape).cuda()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]) :
                    if j < k :
                        mask[i,j,k] = -np.inf
        masked = [ms + mask for ms in mm_scaled]  # headnum * batchsize * sent length * sent length
        softmaxed = [sm(x) for x, sm in zip(masked, self.softmax1)] # headnum * batchsize * sent length * sent length
        attention = [torch.matmul(sm, v) for sm, v in zip(softmaxed, Vs)] # headnum * batchsize * sent length * (embedding dim / headnum)
        return torch.cat(attention, dim=2) # batchsize * sent length * embedding dim

    def add_norm1(self, attention, x):
        do = self.dec_dropout1(attention) # batchsize * sent length * embedding dim
        x_do = x + do   # batchsize * sent length * embedding dim
        return self.dec_norm1(x_do)

    def multi_head_attention(self, Q, K, V):
        Qs = [wq(Q) for wq in self.wqs]  # headnum * batchsize * sent length * (embedding dim / headnum)
        Ks = [wk(K) for wk in self.wks]  # headnum * batchsize * sent length * (embedding dim / headnum)
        Vs = [wv(V) for wv in self.wvs]  # headnum * batchsize * sent length * (embedding dim / headnum)

        mm_scaled = [torch.matmul(Q, K.transpose(1, 2))/(self.config.emb_dim/self.config.headnum)**(1/2)
                     for Q, K in zip(Qs, Ks)]          # headnum * batchsize * sent length * sent length
        softmaxed = [sm(x) for x, sm in zip(mm_scaled, self.softmax2)] # headnum * batchsize * sent length * sent length
        attention = [torch.matmul(sm, v) for sm, v in zip(softmaxed, Vs)] # headnum * batchsize * sent length * (embedding dim / headnum)
        return torch.cat(attention, dim=2) # batchsize * sent length * embedding dim

    def add_norm2(self, sublayer, x):
        do = self.dec_dropout2(sublayer) # batchsize * sent length * embedding dim
        x_do = x + do   # batchsize * sent length * embedding dim
        return self.dec_norm2(x_do)

    def add_norm3(self, fc1, x):
        do = self.dec_dropout3(fc1) # batchsize * sent length * embedding dim
        x_do = x + do   # batchsize * sent length * embedding dim
        return self.dec_norm3(x_do)

    def forward(self, dec_in, enc_out):
        attention_Q, attention_K, attention_V = dec_in, dec_in, dec_in
        masked_attention = self.masked_multi_head_attention(attention_Q, attention_K, attention_V) # batchsize * sent length * embedding dim
        add_norm1 = self.add_norm1(masked_attention, dec_in) # batchsize * sent length * embedding dim

        Q, K, V = add_norm1, enc_out, enc_out
        attention = self.multi_head_attention(Q, K, V)
        add_norm2 = self.add_norm2(attention, Q)

        fc1 = self.fc(add_norm2) # batchsize * sent length * embedding dim
        out = self.add_norm3(fc1, add_norm2) # batchsize * sent length * embedding dim
        return out