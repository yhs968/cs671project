import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class EncoderRNN(nn.Module):
    def __init__(self, shared_emb, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.vocab_size = shared_emb.weight.size(0)
        self.emb_size = shared_emb.weight.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = self.init_emb(shared_emb)
        self.gru = nn.GRU(self.emb_size, hidden_size, num_layers, bidirectional = True)
        
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, hidden):
        '''
        Args:
            input (seq_len, batch_size): input sentence
        '''
        embedded = self.emb(input)
        outputs, hidden = self.gru(embedded, hidden)

        return outputs, hidden
    
    def init_emb(self, shared_emb):
        if shared_emb == None:
            return nn.Embedding(self.vocab_size, self.emb_size)
        else:
            return shared_emb
    
    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
        
        return h0

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size))
        std = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-std, std)

    def forward(self, decoder_hidden, encoder_output):
        '''
        Notations:
            L: num_layers=1
            B: batch_size
            H: hidden_size=input_hidden_size x 2
            T: input_seq_len
        
        Args:
            decoder_hidden (L,B,H)
            encoder_output (T,B,H)
            
        Returns:
            attn_energies (B,T)
        '''
        input_seq_len = encoder_output.size(0)
        batch_size = encoder_output.size(1)

        # a_ij (B,T): attention weights for a given timestep i.
        attn_energies = Variable(torch.zeros(batch_size, input_seq_len))
        encoder_output = encoder_output.transpose(0,1) #(B,T,H)
        decoder_hidden = decoder_hidden.transpose(0,1).expand_as(encoder_output)
        attn_energies = self.score(decoder_hidden, encoder_output)
        
        # Normalize and resize
        attn_energies = F.softmax(attn_energies).unsqueeze(1) #(B,1,T)
        
        return attn_energies

    def score(self, decoder_hidden, encoder_output) :
        '''
        Args:
            decoder_hidden (B,T,H)
            encoder_output (B,T,H)
            
        Returns:
            energy (B,T)
        '''
        energy = self.linear(torch.cat((decoder_hidden, encoder_output), dim=2)) #(B,T,H)
        energy = energy.transpose(2,1) #(B,H,T)
        v = self.v.repeat(energy.size(0),1).unsqueeze(1) #(B,1,H)
        energy = torch.bmm(v, energy) #(B,1,T)
        energy = energy.squeeze(1) #(B,T)
        
        return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, shared_emb, hidden_size, num_layers):
        super().__init__()
        self.vocab_size = shared_emb.weight.size(0)
        self.emb_size = shared_emb.weight.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.emb = self.init_emb(shared_emb)
        self.gru = nn.GRU(hidden_size+self.emb_size, hidden_size, num_layers)
        self.attn = Attn(hidden_size)
        self.out = nn.Linear(hidden_size, self.vocab_size)
        
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, decoder_input, decoder_hidden, encoder_output):
        '''
        Notations:
            S: seq_len=1
            B: batch_size
            H: hidden_size=input_hidden_size x 2
            T: input_seq_len
            d: emb_size
            L: num_layers=1
            V: vocabulary size
            
        Args:
            decoder_input (S,B): a word index for current time step
            decoder_hidden (L,B,H)
            encoder_output (T,B,H)
        
        Returns:
            decoder_output (B,V)
            decoder_hidden (L,B,H)
        '''
        # Get the embedding of the current input word(not a sentence)
        batch_size = decoder_input.size(1)
        embedded = self.emb(decoder_input) # (S,B,d)
        
        # calculate a_ij for all j's
        attn_weights = self.attn(decoder_hidden, encoder_output) # (B,S,T)
        encoder_output = encoder_output.transpose(0, 1) #(B,T,H)
        
        # calculate the context vector
        context = torch.bmm(attn_weights, encoder_output) #(B,S,H)
        context = context.transpose(0, 1) # (S,B,H)
        
        # input for the gru
        gru_input = torch.cat((embedded, context), 2) # (S,B,d+H)
#         print(gru_input.size())
        
        # decoder_output: (S,B,H)
        decoder_output, decoder_hidden = self.gru(gru_input, decoder_hidden)
        decoder_output = decoder_output.squeeze(0) # (B,H)
        decoder_output = self.out(decoder_output) # (B,V)

        return decoder_output, decoder_hidden, attn_weights
    
    def init_emb(self, shared_emb):
        if shared_emb == None:
            return nn.Embedding(self.vocab_size, self.emb_size)
        else:
            return shared_emb
    
    def init_hidden(self, batch_size):
        '''
        Args:
            B (int)
            
        Returns:
            s0 (L,B,H)
        '''
        s0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
        if torch.cuda.is_available():
            s0 = s0.cuda()
            
        return s0