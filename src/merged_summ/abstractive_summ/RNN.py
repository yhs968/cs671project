import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class EncoderRNN(nn.Module):
	def __init__(self, n_input_words, dim_embedding, dim_hidden, n_layers, dropout):
		super(EncoderRNN, self).__init__()
		self.n_input_words = n_input_words
		self.dim_embedding = dim_embedding
		self.dim_hidden = dim_hidden
		self.n_layers = n_layers
		self.dropout = dropout

		self.embedding = nn.Embedding(n_input_words, dim_embedding)
		self.gru = nn.GRU(dim_embedding, dim_hidden, n_layers, dropout = dropout, bidirectional = True)
		
	def forward(self, input_seqs, input_lengths, hidden):
		embedded = self.embedding(input_seqs)
		packed = pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, output_lengths = pad_packed_sequence(outputs)
		outputs = outputs[:, :, :self.dim_hidden] + outputs[:, : ,self.dim_hidden:] # Sum bidirectional outputs

		return outputs, hidden


class Attn(nn.Module):
	def __init__(self, method, dim_hidden):
		super(Attn, self).__init__()
		
		self.method = method
		self.dim_hidden = dim_hidden
		
		if self.method == 'general':
			self.attn = nn.Linear(self.dim_hidden, self.dim_hidden)

	def forward(self, hidden, encoder_outputs):
		max_len = encoder_outputs.size(0)
		batch_size = encoder_outputs.size(1)

		# Create variable to store attention energies
		attn_energies = Variable(torch.zeros(max_len, batch_size)).cuda() # S x B

		for i in range(max_len) : 
			attn_energies[i] = self.score(hidden.squeeze(0), encoder_outputs[i])

		# Normalize energies to weights in range 0 to 1, resize to B x 1 x S
		return F.softmax(attn_energies.transpose(0, 1)).unsqueeze(1)

	def score(self, hidden, encoder_output) :
		# hidden : B x dim, encoder_output : B x dim
		if self.method == 'dot' :
			energy = torch.bmm(hidden.unsqueeze(1), encoder_output.unsqueeze(2)).squeeze(1) 

		elif self.method == 'general' :
			energy = self.attn(encoder_output)
			energy = torch.bmm(hidden.unsqueeze(1), energy.unsqueeze(2)).squeeze(1)
		
		return energy


class AttnDecoderRNN(nn.Module):
	def __init__(self, n_output_words, dim_embedding, dim_hidden, attn_model, n_layers, dropout):
		super(AttnDecoderRNN, self).__init__()
		self.n_output_words = n_output_words
		self.dim_embedding = dim_embedding
		self.dim_hidden = dim_hidden
		self.n_layers = n_layers
		self.dropout = dropout
		self.attn_model = attn_model

		self.embedding = nn.Embedding(n_output_words, dim_embedding)
		self.gru = nn.GRU(dim_embedding, dim_hidden, n_layers, dropout = dropout)
		self.concat = nn.Linear(dim_hidden * 2, dim_hidden)
		self.out = nn.Linear(dim_hidden, n_output_words)
		self.attn = Attn(attn_model, dim_hidden)

	def forward(self, input_seq, last_hidden, encoder_outputs):
		# Get the embedding of the current input word (last output word)
		batch_size = input_seq.size(0)
		embedded = self.embedding(input_seq)
		embedded = embedded.view(1, batch_size, self.dim_embedding) # 1 x B x N

		rnn_output, hidden = self.gru(embedded, last_hidden)

		attn_weights = self.attn(rnn_output, encoder_outputs)
		# apply to encoder outputs to get weighted average
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # (B x 1 x S) x (B x S x dim) -> B x 1 x dim

		# Attentional vector using the RNN hidden state and context vector concatenated together
		rnn_output = rnn_output.squeeze(0)	# B x dim
		context = context.squeeze(1)		# B x dim 
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = F.tanh(self.concat(concat_input))
		output = self.out(concat_output)

		return output, hidden, attn_weights
