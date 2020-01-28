import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import numpy as np

class Trainer():
    def __init__(self, config, dataset, model, evaluator):
        self.config = config
        self.dataset = dataset
        self.test_gen = iter(dataset.TEST)
        self.evaluator = evaluator
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

        if self.config.hard_labels :
            criterion = nn.CrossEntropyLoss()
        else :
            criterion = smooth_label_loss

        optimizer = optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=self.config.learning_rate)
        dynamic_lr = lambda step :  self.config.lr_scale * (self.config.emb_dim)**(-1/2) * \
                                   min((step+1)**(-1/2), (step+1) * self.config.warmup_steps**(-3/2))
        if self.config.use_dynamic_lr:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,dynamic_lr)

        if self.config.reset_model :
            tr_losses, te_losses, tr_bleus, te_bleus = [], [], [], []
        else :
            with open('./out/losses_{}.pkl'.format(self.config.model_name), 'rb') as f :
                tr_losses, te_losses, tr_bleus, te_bleus = pickle.load(f)
            f.close()

        steps = 0
        for epoch in range(self.config.epoch_num) :
            for idx, batch in enumerate(self.dataset.TRAIN) :
                fr = batch.fr  # sentence length * batch size
                en = batch.en  # sentence length * batch size

                # self.model(src, trg)
                out = self.model(fr, en)  # batch size * sentence length * vocab size
                out = out.transpose(1,2)  # batch size * vocab size * sentence length
                tgt = en.transpose(0,1)  # batch size * sentence length

                optimizer.zero_grad()
                loss = criterion(out, tgt)
                loss.backward()

                if self.config.use_dynamic_lr :
                    scheduler.step()
                optimizer.step()

                # train evaluation
                tr_bleu = self.evaluator.evaluate(epoch, idx, steps, self.model, fr, en, out,
                                                  loss.item(), optimizer, mode='TRAIN')

                # a rather rough way to keep the test set generator going without StopIteration
                try :
                    te_bat = next(self.test_gen)
                except StopIteration:
                    self.test_gen = iter(self.dataset.TEST)
                    te_bat = next(self.test_gen)

                # test evaluation
                if steps % self.config.test_step == 0 :
                    te_loss, te_bleu = self.evaluator.evaluate(None, None, steps, self.model, te_bat.fr, te_bat.en,
                                                               None, criterion, None, mode='TEST')
                    tr_losses.append(loss.item())
                    te_losses.append(te_loss)
                    tr_bleus.append(tr_bleu)
                    te_bleus.append(te_bleu)
                    self.plot_figures(tr_losses, te_losses, tr_bleus, te_bleus)

                # save models and plot losses
                if steps % self.config.save_step == self.config.save_step-1 :
                    torch.save(self.model, './out/{}.mdl'.format(self.config.model_name))
                    with open('./out/losses_{}.pkl'.format(self.config.model_name),'wb') as f :
                        pickle.dump([tr_losses, te_losses, tr_bleus, te_bleus], f)
                    f.close()

                steps += 1

                # early stop
                converged = self.early_stop(te_losses)
                if converged :
                    break
            if converged:
                print('Model Converged...')
                break

    def early_stop(self, losses):
        if len(losses) > 100 :
            if np.mean(losses[-50:]) < np.mean(losses[-20:]) :
                return True
            else :
                return False

    def plot_figures(self, tr_losses, te_losses, tr_bleus, te_bleus):
        # Loss
        plt.plot(tr_losses)
        plt.plot(te_losses)
        plt.title('Train & Test Loss Curve')
        plt.xlabel('Steps [ every {} step ]'.format(self.config.test_step))
        plt.ylabel('Loss')
        plt.legend(['train','test'])
        plt.savefig('./img/loss_{}.png'.format(self.config.model_name))
        plt.close()

        # BLEU score
        plt.plot(tr_bleus)
        plt.plot(te_bleus)
        plt.title('Train & Test BLEU Score')
        plt.xlabel('Steps [ every {} step ]'.format(self.config.test_step))
        plt.ylabel('BLEU Score')
        plt.legend(['train','test'])
        plt.savefig('./img/bleu_{}.png'.format(self.config.model_name))
        plt.close()
