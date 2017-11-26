import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SentenceEncoder(nn.Module):
    
    def __init__(self, vocab_size, emb_size, n_kernels, kernel_sizes, pretrained = None, static = False):
        '''
        Args:
            vocab_size (int): size of the vocabulary
            emb_size (int): dimension of word embeddings
            n_kernels (int): the number of filters
            kernel_sizes (int): a list of sliding windows to be used
            static (bool): whether you want the embeddings to be updated or not
        '''
        super().__init__()
        in_channels = 1
        self.vocab_size = vocab_size
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.init_emb(pretrained)
        if static:
            self.emb.weight.requires_grad = False
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, n_kernels, (h, emb_size))
             for h in kernel_sizes])
        
        if torch.cuda.is_available():
            self.cuda()
    
    def init_emb(self, emb_pretrained):
        if emb_pretrained == None:
            return
        else:
            self.emb.weight = nn.Parameter(emb_pretrained.weight.data)

    def forward(self, s):
        '''
        Args:
            s (seq_len): a sentence of type torch.LongTensor.
            Each entries represent a word index.
        '''
        # (batch_size = 1, in_channel, seq_len, emb_size)
        s = self.emb(s).unsqueeze(1)
        
        feature_map = [F.relu(conv(s)).squeeze(3)
                       for conv in self.convs]
        feature_pooled = [F.max_pool1d(c, c.size(2)).squeeze(2)
                          for c in feature_map]
        feature_pooled = torch.cat(feature_pooled, 1)
        
        return feature_pooled


class DocumentEncoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, s, h, c):
        '''
        Args:
            s: a batch of fixed-length sentences. (seq_len) x (batch_size) x (output_size)
            h: previous hidden state. (num_layers) x (batch_size) x (hidden_size)
            c: previous cell state. (num_layers) x (batch_size) x (hidden_size)
        '''
        output,(h,c) = self.lstm(s,(h,c))
        return output,h,c
        
    def init_h0c0(self, batch_size = 1):
        # dimension: num_layers*num_directions, batch_size, hidden_size
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        
        return h0,c0
    
class ExtractorCell(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        # arguments
        self.input_size = input_size 
        self.hidden_size = hidden_size
        
        # layers and operations
        self.lstmc = nn.LSTMCell(input_size, hidden_size)
        self.h2p = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, s, h, h_, c_, p):
        '''
        s: s_{t-1} (batch=1, input_size)
        h: h_t (batch=1, hidden_size)
        h_: hbar_{t-1} (batch=1, hidden_size)
        c_: cbar_{t-1} (batch=1, hidden_size)
        p: p_{t-1}. (batch=1, 1)
        '''

        s_weighted = p.expand_as(s) * s
        h_, c_ = self.lstmc(s_weighted, (h_,c_))
        # (batch, hidden_size*2)
        h_cat = torch.cat([h, h_], dim = 1)
        
        batch_size = h_cat.size(0)
        logit = Variable(torch.zeros(batch_size, 1))
        if torch.cuda.is_available():
            logit = logit.cuda()
            
        for b in range(batch_size):
            logit[b] = self.h2p(h_cat[b])
        p = self.sigmoid(logit)
        
        if torch.cuda.is_available():
            h_ = h_.cuda()
            c_ = c_.cuda()
            p = p.cuda()
        
        return h_, c_, p
    
    def init_p(self, h0, hn):
        batch_size = h0.size(0)
        h_cat = torch.cat([h0, hn], dim = 1)
        logit = Variable(torch.zeros(batch_size, 1))
        
        if torch.cuda.is_available():
            logit = logit.cuda()
            
        for b in range(batch_size):
            logit[b] = self.h2p(h_cat[b])
            
        p0 = self.sigmoid(logit)
        
        if torch.cuda.is_available():
            p0 = p0.cuda()
        
        return p0
    
class DocumentClassifier(nn.Module):
    
    def __init__(self, sent_size, n_classes):
        super().__init__()
        self.sent_size = sent_size
        self.n_classes = n_classes
        self.linear = nn.Linear(sent_size, n_classes)
        
        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, p, s):
        '''
        Args:
            p (seq_len, ): extraction probability
            s (seq_len, sent_size): encoded sentences
        
        Returns:
            q: (n_classes, 1): document classification probability
        '''
        s_avg = torch.sum(p.expand_as(s) * s, 0)
        s_avg /= torch.sum(p)
        
        logit = self.linear(s_avg) #(seq_len, n_classes)
        q = F.log_softmax(logit)
        
        return q