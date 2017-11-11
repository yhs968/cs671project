import torch
import torch.nn as nn
from torch.autograd import Variable

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