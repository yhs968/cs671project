from nltk import sent_tokenize, wordpunct_tokenize
from collections import Counter
from itertools import chain
import torch
import numpy as np

class Vocab:
    '''Abstract vocabulary class that has useful helper functions
    '''
    def __init__(self, corpus, top_k = None, case_sensitive = False):
        '''
        Builds a vocabulary, using the words in the corpus

        Args:
            corpus: string of text.
            top_k: the number of words to be included in the vocabulary, including the special tokens:
            "UNK(unknown)", "_BEGIN_(beginning of a sentence)", "_END_(end of a sentence)"

        Returns:
            word2id: dictionary of (word, id) pairs
            id2word: list of words in the vocabulary
        '''
        if type(top_k) == int:
            top_k -= 3
        if not case_sensitive:
            corpus = corpus.lower()
        
        word_counts = Counter(wordpunct_tokenize(corpus)).most_common(top_k)

        id2word = sorted([word for word,count in word_counts]) + ['UNK','_BEGIN_','_END_']
        word2id = {word: i for i, word in enumerate(id2word)}

        self.id2word = id2word
        self.word2id = word2id
        self.V = len(id2word)

    def sents2id(self, text, top_k = None, case_sensitive = False):
        '''Tokenizes a text into sentences, mapping the words to corresponding indices.

        Args:
            text: string.

        Returns:
            sents_list: List of sentences, where each sentences are the list of word indices.
        '''
        word2id = self.word2id
        id2word = self.id2word
        
        if not case_sensitive:
            text = text.lower()
        
        sents = sent_tokenize(text)

        sents_list = []
        for i in range(len(sents)):
            sent = wordpunct_tokenize(sents[i])
            sent = [word2id[word] if word in word2id else word2id['UNK'] for word in sent]
            sent = [word2id['_BEGIN_']] + sent + [word2id['_END_']]
            sents_list.append(sent)

        return sents_list

    def id2sents(self, sents):
        '''Returns the string representation of the sentences, where sentences is a list of sentences
        and each sentences are lists of word ids.

        Args:
            sents: a list of word ids in the dictionary

        Returns:
            sents_str: string representation of sentences.
        '''

        return ' '.join([self.id2word[i_word] for i_word in chain(*sents)])

    def sent2onehot(self, tokens):
        '''
        Converts the list of word indices into the corresponding list of one-hot vectors

        Args:
            tokens: a sequence of word indices

        Returns:
            onehots: a sequence of one-hot vectors corresponding to tokens.
        '''
        onehots = []
        for i in tokens:
            vec = np.zeros(shape=self.V, dtype=int)
            vec[i] = 1
            onehots.append(vec)

        onehots = torch.from_numpy(np.vstack(onehots)).type(torch.FloatTensor)
        return onehots
    
    def sent2emb(self, tokens, emb):
        '''
        Converts the list of word indices into the corresponding list of embeddings

        Args:
            tokens: a sequence of word indices
            emb: pretrained embeddings. nn.Embedding type.

        Returns:
            embedded: a sequence of embeddings corresponding to tokens.
        '''
        embedded = [emb.weight.data[i].view(1,-1) for i in tokens]
        embedded = torch.cat(embedded, dim = 0)
        return embedded

    def onehot2sent(self, vecs):
        '''
        Converts a sequence of one-hot vectors into a sequence of word indices

        Args:
            vecs: a sequence of one-hot vectors. 
            Should be a torch tensor where each rows correspond to a one-hot vector of a word.

        Returns:
            sent: a list of word indices that corresponds to vecs
        '''
        maxs, argmaxs = torch.max(vecs, dim = 1) # dim: axis to get argmaxs

        sent = [self[i] for i in argmaxs]
        return sent
    
    def __str__(self):
        return str(self.word2id)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.id2word[key]
        elif type(key) == str:
            return self.word2id[key]
        else:
            # print(key)
            print('Wrong type')
            return None