import os
from os.path import join
from collections import Counter
from itertools import chain
from nltk import sent_tokenize, wordpunct_tokenize
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class DocumentDataset(Dataset):
    '''
    Documents dataset.
    '''
    
    def __init__(self, filename, vocab, case_sensitive = False):
        '''
        Args:
            filename (string): full path of the document file
            vocab (Vocab): Vocabulary class that contains the vocabulary for a corpus
            emb (nn.Embedding): word embeddings corresponding to the words in words_dict
            case_sensitive (bool): whether lower/uppercase letters differ
        '''
        
        with open(filename) as f:
            raw = f.read()
        if not case_sensitive:
            raw = raw.lower()
        
        self.vocab = vocab
        # input sentences
        self.inputs = vocab.sents2id(raw, case_sensitive)
        np.random.seed(0)
        self.targets = [np.random.randint(2) for sent in self.inputs]
        
    def __getitem__(self, idx):
        inputs = torch.LongTensor(self.inputs[idx]) 
        targets = torch.LongTensor([self.targets[idx]])
        
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)