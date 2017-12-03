import math
import random
import time

import RNN

import torch
import torch.nn as nn
from torch import optim
from rouge import Rouge

import matplotlib
matplotlib.use('Agg')

import train
import lib
import evaluate

random.seed(1234)
torch.manual_seed(1234)

rouge = Rouge()

MIN_INPUT_SEQ_LENGTH = 1
MAX_INPUT_SEQ_LENGTH = math.inf
MAX_GEN_SEQ_LENGTH = 100
MIN_WORD_CNT = 1 
train_ratio = 0.99 
valid_ratio = 0.005

# Configure models
attn_model = 'general'
dim_embedding = 500
dim_enc_hidden = 500
dim_dec_hidden = 500
n_rnn_layers = 1
dropout = 0
batch_size = 100
beam_size = 20

# Configure training/optimization
max_gradient_norm = 50.0
n_epochs = 100
batch_print_every = 100
epoch_print_every = 1
evaluate_every = 1
plot_every = 1

# prepare dataset
input_lang, output_lang, pairs = lib.prepare_data('data/eng-fra.txt', 'eng', 'fra', MIN_INPUT_SEQ_LENGTH, MAX_INPUT_SEQ_LENGTH, MIN_WORD_CNT, False)
idxs_pairs = lib.sentence_to_idxs(input_lang, output_lang, pairs)
train_set, val_set, test_set = lib.split_train_val_test(idxs_pairs, train_ratio, valid_ratio)

# Initialize models
encoder = RNN.EncoderRNN(input_lang.n_words, dim_embedding, dim_enc_hidden, n_rnn_layers, dropout).cuda()
decoder = RNN.AttnDecoderRNN(output_lang.n_words, dim_embedding, dim_dec_hidden, attn_model, n_rnn_layers, dropout).cuda()

# Initialize optimizers
encoder_optimizer = optim.Adam(encoder.parameters())
decoder_optimizer = optim.Adam(decoder.parameters())

batch_train_losses = []
epoch_train_losses = []
epoch_val_scores = []
epoch_test_scores = []

epoch_start = time.time()
epoch_loss_total = 0
for epoch in range(n_epochs) : 
	epoch += 1
	
	batched_train_set = lib.get_batched_set(train_set, batch_size, True)
	batch_start = time.time()
	batch_loss_total = 0
	
	for i in range(len(batched_train_set)) : 
		input_batch, input_lengths, target_batch, target_lengths = lib.idxs_to_tensor(input_lang, output_lang, batched_train_set[i])
		loss, ec, dc = train.train(input_batch, input_lengths, target_batch, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, max_gradient_norm)

		batch_loss_total += loss
		epoch_loss_total += loss

		if (i+1) % batch_print_every == 0 : 
			batch_loss_avg = batch_loss_total / batch_print_every
			batch_cur = time.time()
			print('=====> %s (%d %d%%) loss : %.4f' % (lib.as_minutes(batch_cur - batch_start), (i+1), (i+1)/len(batched_train_set)*100, batch_loss_avg))
			batch_loss_total = 0
			batch_start = time.time()
	
	epoch_train_losses.append(epoch_loss_total / len(batched_train_set))
	epoch_loss_total = 0

	if epoch % epoch_print_every == 0 :
		epoch_cur = time.time()
		print('%d epochs(%d%%) %s' % (epoch, epoch/n_epochs*100, lib.as_minutes(epoch_cur - epoch_start)))
		print()
		epoch_start = time.time()
	
	if epoch % evaluate_every == 0 :
		# evaluate on validation set
		epoch_val_score_total = 0
		batched_val_set = lib.get_batched_set(val_set, batch_size, False)
		for i in range(len(batched_val_set)) : 
			input_batch, input_lengths, target_batch, target_lengths = lib.idxs_to_tensor(input_lang, output_lang, batched_val_set[i])
			epoch_val_score_total += evaluate.evaluate(input_batch, input_lengths, target_batch, MAX_GEN_SEQ_LENGTH, beam_size, encoder, decoder, output_lang, 1, rouge)
		epoch_val_scores.append(epoch_val_score_total / len(batched_val_set))

		# evaluate on test set
		epoch_test_score_total = 0
		batched_test_set = lib.get_batched_set(test_set, batch_size, False)
		for i in range(len(batched_test_set)) : 
			input_batch, input_lengths, target_batch, target_lengths = lib.idxs_to_tensor(input_lang, output_lang, batched_test_set[i])
			epoch_test_score_total += evaluate.evaluate(input_batch, input_lengths, target_batch, MAX_GEN_SEQ_LENGTH, beam_size, encoder, decoder, output_lang, 0, rouge)
		epoch_test_scores.append(epoch_test_score_total / len(batched_test_set))

	if epoch % plot_every == 0 :
		lib.plot_epochs(epoch_train_losses, epoch_val_scores, epoch_test_scores, 'plotting')
