import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

class Trainer():
    def __init__(self, config, dataset, model):
        self.config = config
        self.dataset = dataset
        self.model = model
        print('\n\n')

    def train(self):
        def smooth_label_loss(pred, target):
            # reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
            target = target.unsqueeze(1)
            eps = self.config.smooth_rate
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, target, 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
            return loss

        dynamic_lr = lambda step :  self.config.lr_scale * (self.config.emb_dim)**(-1/2) * \
                                   min((step+1)**(-1/2), (step+1) * self.config.warmup_steps**(-3/2))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=self.config.learning_rate)
        if self.config.use_dynamic_lr:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,dynamic_lr)

        if self.config.reset_model :
            losses = []
        else :
            with open('./out/losses.pkl', 'rb') as f : losses = pickle.load(f)
            f.close()

        steps = 0
        for epoch in range(self.config.epoch_num) :
            for idx, batch in enumerate(self.dataset.TRAIN) :
                fr = batch.fr  # sentence length * batch size
                en = batch.en  # sentence length * batch size

                # self.model(input, target)
                out = self.model(fr, en)  # batch size * sentence length * vocab size
                out = out.transpose(1,2)  # batch size * vocab size * sentence length
                pred = torch.argmax(out, dim=1)
                en = en.transpose(0,1)  # batch size * sentence length

                optimizer.zero_grad()
                if self.config.hard_labels :
                    loss = criterion(out, en)
                else :
                    loss = smooth_label_loss(out, en)
                loss.backward()
                losses.append(loss.item())

                if self.config.use_dynamic_lr :
                    scheduler.step()
                optimizer.step()

                print('EPOCH {0} [ batch {1} ] : LOSS = {2:.5f} | LR = {3}'
                      .format(epoch+1, idx+1, loss.item(), optimizer.param_groups[0]['lr']))

                if steps % self.config.save_step == self.config.save_step-1 :
                    torch.save(self.model, './out/{}.mdl'.format(self.config.model_name))
                    with open('./out/losses.pkl','wb') as f : pickle.dump(losses, f)
                    f.close()
                if steps % self.config.print_step == 0 :
                    self.print_example(fr.transpose(0,1), en, pred)
                    self.plot_figures(losses)
                steps += 1

    def plot_figures(self, losses):
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.savefig('./img/loss.png')
        plt.close()

    def print_example(self, fr, en, pred):
        print('\n\n\n### EXAMPLE ###')
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