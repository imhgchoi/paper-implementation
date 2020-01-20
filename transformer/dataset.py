import numpy as np
import pandas as pd
from torchtext.data import Field, TabularDataset, Iterator
from sklearn.model_selection import train_test_split
import pdb

class Dataset():
    def __init__(self, config):
        self.config = config
        self.raw_en = self.load(self.config.en_dir)
        self.raw_fr = self.load(self.config.fr_dir)
        self.save_as_csv()
        self.TRAIN, self.TEST = self.tokenize()

    def load(self, filedir):
        with open(filedir, mode='rt', encoding='utf-8') as f:
            if self.config.debug :
                data = [next(f).replace('\n','') for _ in range(self.config.debug_num)]
            else :
                data = [next(f).replace('\n','') for _ in range(self.config.data_num)]
        f.close()
        return data

    def save_as_csv(self):
        pair_df = pd.DataFrame(list(zip(self.raw_en, self.raw_fr)), columns= ['en','fr'])
        train, test = train_test_split(pair_df, test_size=0.15, random_state=self.config.seed)

        print('English-French Pair Size : {}'.format(len(pair_df)))
        print('Train Set Size : {}'.format(len(train)))
        print('Test Set Size : {}'.format(len(test)))

        train.to_csv('./data/train.csv', encoding='utf-8', index=False, header=False)
        test.to_csv('./data/test.csv', encoding='utf-8', index=False, header=False)

    def tokenize(self):
        ENGLISH = Field(sequential=True,
                        use_vocab=True,
                        tokenize=str.split,
                        lower=True,
                        init_token="<sos>",
                        eos_token="<eos>")
        FRENCH = Field(sequential=True,
                        use_vocab=True,
                        tokenize=str.split,
                        lower=True,
                        init_token="<sos>",
                        eos_token="<eos>")

        """
        in order for this to work, change
        "csv.field_size_limit(sys.maxsize)" in torchtext/utils.py to "csv.field_size_limit(maxInt)"
        """
        train, test = TabularDataset.splits(path='./data/', train='train.csv', test='test.csv',
                                            format='csv', fields=[('en',ENGLISH),('fr',FRENCH)])
        ENGLISH.build_vocab(train, test)
        FRENCH.build_vocab(train, test)
        self.en_vocabsize = len(ENGLISH.vocab)
        self.fr_vocabsize = len(FRENCH.vocab)

        if self.config.debug :
            train_loader, test_loader = Iterator.splits((train, test), batch_size=2, device="cuda", repeat=False)
        else :
            train_loader, test_loader = Iterator.splits((train, test), batch_size=self.config.batchsize, device="cuda", repeat=False)
        return train_loader, test_loader