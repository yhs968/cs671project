import logging
import logging, gensim, bz2
from gensim import corpora
import sys
import csv
from modules.data import Documents
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader
import os
import pickle

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
