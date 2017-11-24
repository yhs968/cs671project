import math
import random
import time

import RNN
import lib

import torch
import torch.nn as nn
from torch import optim

from train import *
# from evaluate import *

random.seed(1234)
torch.manual_seed(1234)

MIN_SEQ_LENGTH = 1
MAX_SEQ_LENGTH = math.inf
MIN_WORD_CNT = 1 
train_ratio = 0.6 
valid_ratio = 0.15

# Configure models
attn_model = 'general'
dim_embedding = 500
dim_enc_hidden = 500
dim_dec_hidden = 500
n_rnn_layers = 1
dropout = 0
batch_size = 100

# Configure training/optimization
max_gradient_norm = 50.0
n_epochs = 500
batch_print_every = 10
epoch_print_every = 1
evaluate_every = 10
plot_every = 20

# prepare dataset
input_lang, output_lang, pairs = lib.prepare_data('data/eng-fra.txt', 'eng', 'fra', MIN_SEQ_LENGTH, MAX_SEQ_LENGTH, MIN_WORD_CNT, False)
idxs_pairs = lib.sentence_to_idxs(input_lang, output_lang, pairs)
train_set, val_set, test_set = lib.split_train_val_test(idxs_pairs, train_ratio, valid_ratio)

# Initialize models
encoder = RNN.EncoderRNN(input_lang.n_words, dim_embedding, dim_enc_hidden, n_rnn_layers, dropout).cuda()
decoder = RNN.AttnDecoderRNN(output_lang.n_words, dim_embedding, dim_dec_hidden, attn_model, n_rnn_layers, dropout).cuda()

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters())
decoder_optimizer = optim.Adam(decoder.parameters())
criterion = nn.CrossEntropyLoss()

batch_print_loss_total = 0
epoch_print_loss_total = 0
plot_loss_total = 0
plot_losses = []

ecs = []
dcs = []
eca = 0
dca = 0
epoch_start = time.time()
for epoch in range(n_epochs) : 
	epoch += 1
	
	batched_train_set = lib.get_batched_set(train_set, batch_size, True)
	batch_start = time.time()
	for i in range(len(batched_train_set)) : 
		input_batch, input_lengths, target_batch, target_lengths = lib.idxs_to_tensor(input_lang, output_lang, batched_train_set[i])
		loss, ec, dc = train(input_batch, input_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_gradient_norm)

		# Keep track of loss
		batch_print_loss_total += loss
		epoch_print_loss_total += loss
		plot_loss_total += loss
		eca += ec
		dca += dc

		if (i+1) % batch_print_every == 0 : 
			batch_print_loss_avg = batch_print_loss_total / batch_print_every
			batch_cur = time.time()
			print('=====> %s (%d %d%%) loss : %.4f' % (lib.as_minutes(batch_cur - batch_start), (i+1), (i+1)/len(batched_train_set)*100, batch_print_loss_avg))
			batch_print_loss_total = 0
			batch_start = time.time()

	if epoch % epoch_print_every == 0 :
		epoch_print_loss_avg = epoch_print_loss_total / epoch_print_every
		epoch_cur = time.time()
		print('%d epochs(%d%%) %s' % (epoch, epoch/n_epochs*100, lib.as_minutes(epoch_cur - epoch_start)))
		print_loss_total = 0
		epoch_start = time.time()

	if epoch % evaluate_every == 0 :
		batched_val_set = lib.get_batched_set(val_set, batch_size, False)
		for i in range(len(batched_val_set)) : 
			input_batch, input_lengths, target_batch, target_lengths = lib.idxs_to_tensor(input_lang, output_lang, batched_val_set[i])
#			evaluate(input_batch[0], target_lengths[0])
