import os
from os.path import join
from collections import Counter
from itertools import chain
from nltk import sent_tokenize, wordpunct_tokenize
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import re
from modules.texts import Vocab

class Doc(Dataset):
    '''
    A document datatype
    '''
    def __init__(self, vocab, title, summary, content, labels):
        '''
        Args:
            vocab (Vocab)
            title (str), summary (str), content (str), labels (list of ints)
            input_type (str): type of input key
        '''
        self.vocab = vocab
        self.title = title
        self.summary = summary
        self.content = content
        # input sentences
        self.inputs = vocab.sents2id(content)
        self.targets = labels
        assert len(self.inputs) == len(self.targets)
        
    def __getitem__(self, idx):
        input = torch.LongTensor(list(self.inputs[idx]))
        target = torch.LongTensor([self.targets[idx]])
        
        return input, target
    
    def __len__(self):
        return len(self.inputs)

class DocumentsGroup(Dataset):
    '''
    Documents Group
    '''
    
    def __init__(self, filename, vocab_size = None):
        '''
        Args:
            filename (string): full path of the labeled documents pickle
            vocab_size (int): the size of the vocabulary
            case_sensitive (bool): whether lower/uppercase letters differ
        '''
        
        import pickle
        
        self.filename = filename
        self.vocab_size = vocab_size
        
        self.doc = []
        with open(filename, 'rb') as f:
            dat = pickle.load(f)
        # Build corpus
        corpus = []
        for line in dat:
            tokens = chain(*[wordpunct_tokenize(t) for t in line[:-1]])
            corpus.extend(tokens)
        corpus = ' '.join(corpus)
        
        # Build vocabulary
        self.vocab = Vocab(corpus, top_k = vocab_size)
        
        for line in dat:
            title, summary, content, labels = line
            self.doc.append(Doc(self.vocab, title, summary, content, labels))
        
    def __getitem__(self, idx):
        return self.doc[idx]
    
    def __len__(self):
        return len(self.doc)
    
class GreedyLabeler():
    '''
    Greedy Labeling using ROUGE F-scores
    '''
    
    def __init__(self):
        from rouge import Rouge
        self.rouge = Rouge()
        
    def label(self, reference, corpus, l_type = '1', epsilon = 0.01):
        '''
        Args:
            reference (str): reference summary
            corpus (str): corpus to label
            l_type (str)= label type. 1, 2, or L
            epsilon (float): threshold value for stopping the greedy addition
        '''
        from nltk import sent_tokenize
        
        # Handle label Types
        if type(l_type) == int:
            l_type = str(l_type)
        l_type = l_type.lower()
        
        # Initialize
        ## current summary set
        summary_set = []
        ## candidate sentences
        cand_s = sent_tokenize(corpus)
        ## indices for candidate sentences
        cand_i = set([i for i in range(len(cand_s))])
        
        label = [0 for i in range(len(cand_s))]
        
        max_improvement = 1
        best_score = 0
        
        while len(cand_i) > 0:
            new_summary = [' '.join(summary_set + [sent]) for sent in cand_s]
            # ROUGE scores for each new summaries
            score = [self.rouge.get_scores(reference, s, avg = True)['rouge-%s' % l_type]['f'] for s in new_summary]
            # improvement in ROUGE scores by adding new summaries
            ds = [s - best_score for s in score]
            # best improvement
            max_i, max_improvement = max([(i, d) for i, d in enumerate(ds) if i in cand_i],
                                         key = lambda x:x[1])
            # no more desired improvements
            if max_improvement <= epsilon:
                break
            else:
                label[max_i] = 1
                summary_set.append(cand_s[max_i])
                cand_i.remove(max_i)
                best_score = max_improvement
#                 print(cand_i)
                
        return label
    
class DocumentDataset(Dataset):
    '''
    @deprecated
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