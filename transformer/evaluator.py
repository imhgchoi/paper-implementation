import torch

class Evaluator():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    def evaluate(self, epoch, idx, steps, model, src, tgt, loss, optimizer, mode):
        model.eval()

        if mode == 'TRAIN' :
            print('EPOCH {0} [ batch {1} / {2} ]  TRAIN LOSS = {3:.5f} | LR = {4:.7f}'
                  .format(epoch + 1, idx + 1, len(self.dataset.TRAIN), loss, optimizer.param_groups[0]['lr']))
        elif mode == 'TEST' :
            out = model(src, tgt)
            target = tgt.transpose(0,1)
            l = loss(out.transpose(1,2), target).item()
            print('                             TEST  LOSS = {0:.5f}'.format(l))

            if steps % self.config.print_step == 0:
                out = model(src, tgt)
                pred = torch.argmax(out.transpose(1,2), dim=1)
                self.print_example(src.transpose(0, 1), tgt.transpose(0,1), pred, mode)
            return l


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