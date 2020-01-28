import torch
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

class Evaluator():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def evaluate(self, epoch, idx, steps, model, src, trg, out, loss, optimizer, mode):
        model.eval()

        if mode == 'TRAIN' :
            pred = torch.argmax(out, dim=1)
            bleu = self.compute_BLEU(trg.transpose(0, 1), pred)
            print('EPOCH {0} [ batch {1} / {2} ]  TRAIN LOSS = {3:.5f} | BLEU = {4:.2f} | LR = {5:.7f}'
                  .format(epoch + 1, idx + 1, len(self.dataset.TRAIN), loss, bleu, optimizer.param_groups[0]['lr']))
            return bleu
        elif mode == 'TEST' :
            out = model(src, trg)
            pred = torch.argmax(out.transpose(1,2), dim=1)
            target = trg.transpose(0,1)
            l = loss(out.transpose(1,2), target).item()
            bleu = self.compute_BLEU(target, pred)
            print('\t\t\t   TEST  LOSS = {0:.5f} | BLEU = {1:.2f}'.format(l, bleu))

            if steps % self.config.print_step == 0:
                self.print_example(src.transpose(0, 1), trg.transpose(0,1), pred, mode)
            return l, bleu

    def print_example(self, fr, en, pred, mode):
        print('\n\n\n### {} EXAMPLE ###'.format(mode))
        expl = []
        for output, target, input in zip(pred, en, fr):
            o = ' '.join([self.dataset.en_vocab.vocab.itos[word] for word in output])
            t = ' '.join([self.dataset.en_vocab.vocab.itos[word] for word in target])
            i = ' '.join([self.dataset.fr_vocab.vocab.itos[word] for word in input])
            expl.append([o,t,i])
        print('INPUT : \n',expl[0][2])
        print('\n')
        print('TARGET : \n',expl[0][1])
        print('\n')
        print('OUTPUT : \n',expl[0][0])
        print('\n\n\n')

    def compute_BLEU(self, trg, pred):
        rid = ['<sos>','<eos>','<pad>']
        sent = []
        for prediction, target in zip(pred, trg):
            s = ' '.join([self.dataset.en_vocab.vocab.itos[word] for word in prediction
                          if self.dataset.en_vocab.vocab.itos[word] not in rid])
            t = ' '.join([self.dataset.en_vocab.vocab.itos[word] for word in target
                          if self.dataset.en_vocab.vocab.itos[word] not in rid])
            sent.append([s, t])
        bleu = np.array([sentence_bleu([pair[1].split(' ')], pair[0].split(' '), weights=(0.25,0.25,0.25,0.25))
                         for pair in sent]).mean() * 100
        return bleu
