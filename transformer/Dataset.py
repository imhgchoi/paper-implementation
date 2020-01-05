import numpy as np
import pandas as pd
from torchtext.data import Field, TabularDataset
from sklearn.model_selection import train_test_split
import spacy

class Dataset():
    def __init__(self, config):
        self.config = config
        self.raw_en = self.load(self.config.en_dir)
        self.raw_fr = self.load(self.config.fr_dir)
        self.save_as_csv()
        self.tokenize()

    def load(self, filedir):
        with open(filedir, mode='rt', encoding='utf-8') as f:
            if self.config.debug :
                data = [next(f).replace('\n','').replace('.','') for _ in range(self.config.debug_num)]
            else :
                data = [next(f).replace('\n','').replace('.','')
                        for _ in range(len(open(filedir, mode='rt', encoding='utf-8').readlines()))]
        f.close()
        return data

    def save_as_csv(self):
        pair_df = pd.DataFrame(list(zip(self.raw_en, self.raw_fr)), columns= ['en','fr'])
        print('English-French Pair Size : {}'.format(len(pair_df)))

        train, test = train_test_split(pair_df, test_size=0.15, random_state=self.config.seed)
        print('Train Set Size : {}'.format(len(train)))
        print('Test Set Size : {}'.format(len(test)))

        train.to_csv('./data/train.csv', encoding='utf-8', index=False)
        test.to_csv('./data/test.csv', encoding='utf-8', index=False)

    def tokenize(self):
        ENGLISH = Field(sequential=True,
                        use_vocab=True,
                        tokenize=str.split,
                        lower=True,
                        batch_first=True,
                        init_token="<sos>",
                        eos_token="<eos>")
        FRENCH = Field(sequential=True,
                        use_vocab=True,
                        tokenize=str.split,
                        lower=True,
                        batch_first=True,
                        init_token="<sos>",
                        eos_token="<eos>")

        train, val = TabularDataset.splits(path='./data', train='train.csv', test='test.csv', format='csv',
                                           fields=[('en',ENGLISH),('fr',FRENCH)])
        print(train)