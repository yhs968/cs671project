import unicodedata
import re
import random
import math
import time

import torch
from torch.autograd import Variable


PAD_token = 0 # padding
SOS_token = 1 # start of sequence
EOS_token = 2 # end of sequence


class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
		self.n_words = 3  # Count default tokens

	def index_words(self, sentence):
		for word in sentence.split(' '):
			self.index_word(word)

	def index_word(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

	# Remove words below a certain count threshold
	def trim(self, min_count):
		keep_words = []
		
		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words %s / %s = %.4f' % (len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)))

		# Reinitialize dictionaries
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
		self.n_words = 3 # Count default tokens

		for word in keep_words:
			self.index_word(word)


# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
	s = unicode_to_ascii(s.lower().strip())
	s = re.sub(r"([,.!?])", r" \1 ", s)
	s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
	s = re.sub(r"\s+", r" ", s).strip()
	return s


def read_langs(filename, lang1, lang2, normalize, reverse):
	print("Reading lines...")

	# Read the file and split into lines
	lines = open(filename).read().strip().split('\n')

	# Split every line into pairs and normalize
	if normalize : pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
	else : pairs = [[s for s in l.split('\t')] for l in lines]

	# Reverse pairs, make Lang instances
	if reverse:
		pairs = [list(reversed(p)) for p in pairs]
		input_lang = Lang(lang2)
		output_lang = Lang(lang1)
	else:
		input_lang = Lang(lang1)
		output_lang = Lang(lang2)

	return input_lang, output_lang, pairs


def filter_pairs(pairs, min_length, max_length):
	filtered_pairs = []
	for pair in pairs:
		if len(pair[0]) >= min_length and len(pair[0]) <= max_length and len(pair[1]) >= min_length and len(pair[1]) <= max_length: filtered_pairs.append(pair)
	return filtered_pairs


def prepare_data(filename, lang1, lang2, min_seq_length, max_seq_length, min_word_cnt, reverse):
	input_lang, output_lang, pairs = read_langs(filename, lang1, lang2, True, reverse)
	print("Read %d sentence pairs" % len(pairs))

	pairs = filter_pairs(pairs, min_seq_length, max_seq_length)
	print("Filtered to %d sentence pairs" % len(pairs))

	for pair in pairs:
		input_lang.index_words(pair[0])
		output_lang.index_words(pair[1])
	print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))

	if min_word_cnt > 1 : 
		input_lang.trim(min_word_cnt)
		output_lang.trim(min_word_cnt)

		keep_pairs = []
		for pair in pairs:
			input_sentence = pair[0]
			output_sentence = pair[1]
			keep_input = True
			keep_output = True

			for word in input_sentence.split(' '):
				if word not in input_lang.word2index:
					keep_input = False
					break

			for word in output_sentence.split(' '):
				if word not in output_lang.word2index:
					keep_output = False
					break
	
			# Remove if pair doesn't match input and output conditions
			if keep_input and keep_output:
				keep_pairs.append(pair)

		print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
		pairs = keep_pairs

	return input_lang, output_lang, pairs


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
	return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
	seq += [PAD_token for i in range(max_length - len(seq))]
	return seq


# convert sentence to idxs
def sentence_to_idxs(input_lang, output_lang, pairs) : 
	input_seqs = []
	target_seqs = []

	for i in range(len(pairs)):
		input_seqs.append(indexes_from_sentence(input_lang, pairs[i][0]))
		target_seqs.append(indexes_from_sentence(output_lang, pairs[i][1]))

	idxs_pairs = [[input_seqs[i], target_seqs[i]] for i in range(len(pairs))]

	return idxs_pairs


# split dataset into train, validation, test
def split_train_val_test(pairs, train_ratio, val_ratio) : 
	random.shuffle(pairs)
	n_train = round(train_ratio * len(pairs))
	n_val = round(val_ratio * len(pairs))
	n_test = len(pairs) - (n_train + n_val)

	train_set = [pairs[idx] for idx in range(n_train)]
	val_set = [pairs[idx + n_train] for idx in range(n_val)]
	test_set = [pairs[idx + n_train + n_val] for idx in range(n_test)]

	return train_set, val_set, test_set


# split dataset into batches
def get_batched_set(pairs, batch_size, do_shuffle) : 
	if do_shuffle : random.shuffle(pairs)

	sorted_pairs = sorted(pairs, key=lambda pair : len(pair[0]), reverse = True)

	n_batches = int(len(sorted_pairs) / batch_size)
	batched_pairs = [[pair for pair in sorted_pairs[i * batch_size : (i+1) * batch_size]] for i in range(n_batches)]

	if len(sorted_pairs) % batch_size != 0 : 
		batched_pairs.append([pair for pair in sorted_pairs[n_batches * batch_size : len(pairs)]])

	if do_shuffle : random.shuffle(batched_pairs)

	return batched_pairs


# convert sentence to torch tensor
def idxs_to_tensor(input_lang, output_lang, pairs) : 
	# For input and target sequences, get array of lengths and pad with 0s to max length
	input_lengths = [len(pairs[i][0]) for i in range(len(pairs))]
	input_padded = [pad_seq(pairs[i][0], max(input_lengths)) for i in range(len(pairs))]
	target_lengths = [len(pairs[i][1]) for i in range(len(pairs))]
	target_padded = [pad_seq(pairs[i][1], max(target_lengths)) for i in range(len(pairs))]

	# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
	input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1).cuda()
	target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1).cuda()
	
	return input_var, input_lengths, target_var, target_lengths


# get random batch from the whole pairs
def random_batch(input_lang, output_lang, pairs, batch_size):
	input_seqs = []
	target_seqs = []

	# Choose random pairs, check(should modify to no duplication)
	for i in range(batch_size):
		pair = random.choice(pairs)
		input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
		target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

	# Zip into pairs, sort by length (descending), unzip
	seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
	input_seqs, target_seqs = zip(*seq_pairs)

	# For input and target sequences, get array of lengths and pad with 0s to max length
	input_lengths = [len(s) for s in input_seqs]
	input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
	target_lengths = [len(s) for s in target_seqs]
	target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

	# Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
	input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1).cuda()
	target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1).cuda()
	
	return input_var, input_lengths, target_var, target_lengths


def as_minutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)
