import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import dill

class Trainer():
    def __init__(self, config, loader, evaluator):
        self.config = config
        self.data_loader = loader
        self.evaluator = evaluator
        self.criterion = nn.CrossEntropyLoss()

    def train(self, model):
        if self.config.reset_model :
            tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s = [], [], [], [], [], []
        else :
            with open('./out/perf_idx_{}.pkl'.format(self.config.model_name), 'rb') as f :
                tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s = dill.load(f)
            f.close()

        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)
        step = 0
        for epoch in range(self.config.epoch_num) :
            for batch in self.data_loader :
                step +=1

                model.train()
                out = model(batch)
                optimizer.zero_grad()
                tr_loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask].to(model.device))
                tr_loss.backward()
                optimizer.step()
                tr_losses.append(tr_loss.item())

                # validation
                model.eval()
                out = model(batch)
                vl_loss = self.criterion(out[batch.val_mask], batch.y[batch.val_mask].to(model.device))
                vl_losses.append(vl_loss.item())

                # accuracies
                tr_acc = self.evaluator.get_accuracy(out[batch.train_mask], batch.y[batch.train_mask])
                vl_acc = self.evaluator.get_accuracy(out[batch.val_mask], batch.y[batch.val_mask])
                tr_accs.append(tr_acc)
                vl_accs.append(vl_acc)

                # f1 score
                tr_fscore = self.evaluator.get_fscore(out[batch.train_mask], batch.y[batch.train_mask])
                vl_fscore = self.evaluator.get_fscore(out[batch.val_mask], batch.y[batch.val_mask])
                tr_f1s.append(tr_fscore)
                vl_f1s.append(vl_fscore)

                self.print_current(epoch, tr_loss, vl_loss, tr_acc, vl_acc, tr_fscore, vl_fscore)

                # save model and performance indices
                if step % self.config.save_step == 0 :
                    torch.save(model.state_dict(), './out/{}.mdl'.format(self.config.model_name))
                    with open('./out/perf_idx_{}.pkl'.format(self.config.model_name),'wb') as f :
                        dill.dump([tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s], f)
                    f.close()


                # plot
                if step % self.config.plot_step == 0 :
                    self.plot_figures(tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s)

                # early stop
                converged = self.early_stop(vl_losses)
                if converged :
                    break
            if converged:
                print('Model Converged...')
                self.plot_figures(tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s)
                self.evaluator.test(model, self.data_loader)
                break
        return model

    def early_stop(self, losses):
        if len(losses) > 100 :
            if np.mean(losses[-50:]) < np.mean(losses[-20:]) :
                return True
            else :
                return False

    def print_current(self, epoch, tr_loss, vl_loss, tr_acc, vl_acc, tr_fscore, vl_fscore):
        print("EPOCH {0} ::: TRAIN LOSS = {1:.4f} | ACC = {2:.4f} | F1 = {3:.2f}"
                       " ::: VALID LOSS = {4:.4f} | ACC = {5:.4f} | F1 = {6:.2f}"
              .format(epoch, tr_loss.item(), tr_acc, tr_fscore, vl_loss.item(), vl_acc, vl_fscore))

    def plot_figures(self, tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_fs):
        # Loss
        plt.plot(tr_losses)
        plt.plot(vl_losses)
        plt.title('Train & Val Loss Curve')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend(['train','val'])
        plt.savefig('./img/loss_{}.png'.format(self.config.model_name))
        plt.close()

        # Accuracy
        plt.plot(tr_accs)
        plt.plot(vl_accs)
        plt.title('Train & Val Accuracy Curve')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.legend(['train','val'])
        plt.savefig('./img/acc_{}.png'.format(self.config.model_name))
        plt.close()

        # F1 Score
        plt.plot(tr_f1s)
        plt.plot(vl_fs)
        plt.title('Train & Val F1 Score Curve')
        plt.xlabel('Steps')
        plt.ylabel('F1 Score')
        plt.legend(['train','val'])
        plt.savefig('./img/f1_{}.png'.format(self.config.model_name))
        plt.close()
