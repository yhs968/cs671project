import logging
import logging, gensim, bz2
from gensim import corpora
import sys
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from os.path import join
from collections import Counter
from itertools import chain
from nltk import sent_tokenize, wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
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
    def __init__(self, vocab, title, summary, content, ext_labels, doc_class = None):
        '''
        Args:
            vocab (Vocab)
            title (str), summary (str), content (str), ext_labels (list of ints)
            input_type (str): type of input key
            doc_class (int): type of the whole document
        '''
        self.vocab = vocab
        self.title = title
        self.summary = summary
        self.content = content
        # input sentences
        self.sents = vocab.sents2id(content)
        # extraction labels
        self.ext_labels = ext_labels
        assert len(self.sents) == len(self.ext_labels)
        # document class labels
        self.doc_class = doc_class
        # words
        self.words = list(chain(*vocab.sents2id(self.content)))
        # reference summaries
        self.summ = list(chain(*vocab.sents2id(self.summary)))
        self.mode = 'extract'
        
    def set_mode(self, mode):
        assert mode in ('extract','abstract')
        self.mode = mode
        
    def __getitem__(self, idx):
        if self.mode == 'extract':
            sent = torch.LongTensor(list(self.sents[idx]))
            ext_label = torch.LongTensor([self.ext_labels[idx]])
        
            return sent, ext_label
        
        elif self.mode == 'abstract':
            return torch.LongTensor([self.summ[idx]])
    
    def __len__(self):
        if self.mode == 'extract':
            return len(self.sents)
        elif self.mode == 'abstract':
            return len(self.summ)

class Documents(Dataset):
    '''
    A Set of documents
    '''
    
    def __init__(self, filename, n_samples = None, vocab_size = None):
        '''
        Args:
            filename (string): full path of the extractive labeled documents pickle
            vocab_size (int): the size of the vocabulary
            case_sensitive (bool): whether lower/uppercase letters differ
        '''
        
        import pickle
        
        self.filename = filename
        self.vocab_size = vocab_size
        
        self.doc = []
        with open(filename, 'rb') as f:
            dat = pickle.load(f)
            
        lem = WordNetLemmatizer()
        # Build corpus
        corpus = []
        for i, line in enumerate(dat):
            if n_samples != None and i >= n_samples:
                break
            tokens = chain(*[wordpunct_tokenize(t) for t in line[:-1]])
            # tokens = [lem.lemmatize(t) for t in tokens]
            corpus.extend(tokens)
        corpus = ' '.join(corpus)
        
        # Build vocabulary
        self.vocab = Vocab(corpus, top_k = vocab_size)
        
        # Get Topic Labels
        topics = Topics('./data')
        topics.load()
        topics = topics.topics_top1
        self.dclass2id = {tname: i for i, tname in enumerate(sorted(set(topics)))}
        
        for i, (line, doc_class) in enumerate(zip(dat, topics)):
            if n_samples != None and i >= n_samples:
                break
            title, summary, content, labels = line
            self.doc.append(Doc(self.vocab, title, summary, content, labels, self.dclass2id[doc_class]))
            
    def set_doc_classes(self, doc_classes):
        '''
        Used for document classification
        
        Args:
            doc_classes (list of ints): 
        '''
        assert len(self.doc) == len(doc_classes)
        for d, c in zip(self.doc, doc_classes):
            d.doc_class = c
        
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

class Topics(Dataset):


    def __init__(self, data_dir):
        self.data_dir = data_dir


    def load(self, filename='topics.pkl'):
        self.filename = os.path.join(self.data_dir, filename)
        with open(self.filename, 'rb') as f:
            loaded_file = pickle.load(f)
        self.num_topics = loaded_file["num_topics"]
        self.topics = loaded_file["topics"]
        self.topics_top1 = loaded_file["topics_top1"]

    def run(self, num_topics=6):
        print("Running LDA")
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


        self.num_topics = num_topics

        # Load corpus
        doc_file = os.path.join(self.data_dir, 'kaggle_news_rouge1.pkl')
        docs = Documents(doc_file)
        documents = [docs.doc[idx].content for idx in range(len(docs.doc))]

        # Load stop words
        stoplist = stopwords.words('english')
        stoplist_additional = list('will also said'.split())
        stoplist = stoplist + stoplist_additional

        # Tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(documents)):
            documents[idx] = documents[idx].lower()  # Convert to lowercase.
            documents[idx] = tokenizer.tokenize(documents[idx])  # Split into words.

        # Clear stop words
        texts = [[word for word in document if word not in stoplist] for document in documents]

        #Lemmatize
        lemmatizer = WordNetLemmatizer()
        texts_lemmatized = [[lemmatizer.lemmatize(word) for word in text] for text in texts]
        texts = texts_lemmatized

        #remove one letter words
        texts = [[token for token in text if len(token) > 1] for text in texts]

        # Remove numbers, but not words that contain numbers.
        # texts = [[token for token in text if not token.isnumeric()] for text in texts]

        # remove words that appear only once
        '''
        from collections import defaultdict
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 1]
                 for text in texts]
        from pprint import pprint  # pretty-printer
        '''

        # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
        phrases = Phrases(texts, min_count=20)
        bigram = Phraser(phrases)
        for idx in range(len(texts)):
            for token in bigram[texts[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    texts[idx].append(token)



        # Make dictionary
        dictionary = corpora.Dictionary(texts)
        #filter rare/frequent
        dictionary.filter_extremes(no_below=2, no_above=0.90)
        dictionary.save(os.path.join(self.data_dir,'news_summary.dict'))  # store the dictionary, for future reference

        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.MmCorpus.serialize(os.path.join(self.data_dir,'news_summary.mm'), corpus)
        mm = gensim.corpora.MmCorpus(os.path.join(self.data_dir,'news_summary.mm'))


        # Commence lda
        self.lda = gensim.models.ldamulticore.LdaMulticore(corpus=mm, id2word=dictionary, num_topics=self.num_topics,
            chunksize=500, passes=8, workers=4, eval_every=1, iterations=80)
        self.lda.print_topics(self.num_topics)
        self.topic_dict = {idx:topicstring for idx, topicstring in enumerate(str(input("Input topic names: ")).split())}
        self.topics = [[ [self.topic_dict[lda_topics[0]], float(int(lda_topics[1] * 10000))/10000] for lda_topics in self.lda[doc]] for doc in corpus]
        self.topics_top1 = [doc_topics[np.argmax(np.asarray([a_topic[1] for a_topic in doc_topics]))][0] for doc_topics in self.topics]



        savename = os.path.join(self.data_dir, 'topics.pkl')
        with open(savename, 'bw') as f:
            data_to_dump = {"num_topics":self.num_topics, "topics":self.topics, "topics_top1":self.topics_top1}
            pickle.dump(data_to_dump,f)




    def evaluate(self):
        '''
        for idx in range(10):
            print("Doc {0}: {1}".format(idx, self.topics[idx]))
        '''

        with open(os.path.join(self.data_dir, "topics_ground_truth.yaml"), 'r') as f:
            ground_truth_labels = yaml.load(f)
        counter = 0
        for doc_idx in ground_truth_labels:
            print("Doc {0}: {1}".format(doc_idx, self.topics[doc_idx]))    
            topic_compare = np.argmax(np.asarray([a_topic[1] for a_topic in self.topics[doc_idx]]))
            print(ground_truth_labels[doc_idx].split()[0], self.topics[doc_idx][topic_compare])
            if ground_truth_labels[doc_idx].split()[0] == self.topics[doc_idx][topic_compare][0]:
                counter += 1

        print("Accuracy: {0}".format(float(counter)/float(len(ground_truth_labels))))

'''


    def print_topics(self):
        self.lda.print_topics(num_topics)
'''