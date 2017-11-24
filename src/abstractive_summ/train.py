import lib

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from masked_cross_entropy import *

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_gradient_norm):
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	loss = 0

	# Run words through encoder
	encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

	# Prepare input and output variables for decoder
	batch_size = input_batches.size(1)
	decoder_input = Variable(torch.LongTensor([lib.SOS_token] * batch_size)).cuda()
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	max_target_length = max(target_lengths)
	all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.n_output_words)).cuda()

	# Run through decoder one time step at a time
	for t in range(max_target_length):
		decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_outputs)
		all_decoder_outputs[t] = decoder_output
		decoder_input = target_batches[t]

	# Loss calculation and backpropagation
	loss = masked_cross_entropy(all_decoder_outputs.transpose(0, 1).contiguous(), target_batches.transpose(0, 1).contiguous(), target_lengths)
	loss.backward()
	
	# Clip gradient norms
	ec = clip_grad_norm(encoder.parameters(), max_gradient_norm)
	dc = clip_grad_norm(decoder.parameters(), max_gradient_norm)

	# Update parameters with optimizers
	encoder_optimizer.step()
	decoder_optimizer.step()
	
	return loss.data[0], ec, dc
