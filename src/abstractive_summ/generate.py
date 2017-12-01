import lib
import beam

import torch
from torch.autograd import Variable

def generate(input_batch, input_lengths, max_target_length, beam_size, encoder, decoder, output_lang) :
	encoder.eval()
	decoder.eval()

	# Run through encoder
	encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths, None)

	# Create starting vectors for decoder
	batch_size = input_batch.size(1)
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	max_input_length = input_batch.size(0)

	# beam for each batch
	beam_batch = [[] for i in range(batch_size)]

	# Run through decoder
	for t in range(max_target_length) :
		# for SOS token
		if t == 0 : 
			decoder_input = Variable(torch.LongTensor([lib.SOS_token] * batch_size)).cuda()
			decoder_output, decoder_prob_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
		
			# get top k(beam size) values
			top_values, top_idxs = decoder_prob_output.data.topk(beam_size)
			for batch_idx in range(batch_size) :
				root = beam.BeamTree(0, lib.SOS_token)
				for beam_idx in range(beam_size) :
					log_prob = top_values[batch_idx][beam_idx]
					word_idx = top_idxs[batch_idx][beam_idx]
					beam_batch[batch_idx].append(beam.BeamTree(log_prob, word_idx, root))

		else :
			tmp_beam_batch = [[] for i in range(batch_size)]
			for beam_idx in range(beam_size) : 
				decoder_input = []
				for batch_idx in range(batch_size) : 
					# decoder inputs are words in current beam
					decoder_input.append(beam_batch[batch_idx][beam_idx].word_idx)

				decoder_input = Variable(torch.LongTensor(decoder_input)).cuda()
				decoder_output, decoder_prob_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

				# get top k(beam size) values
				top_values, top_idxs = decoder_prob_output.data.topk(beam_size)
				for batch_idx in range(batch_size) :
					for tmp_beam_idx in range(beam_size) : 

						# if current word is EOS, add it to tmp_beam_batch instead of its children
						if beam_batch[batch_idx][beam_idx].word_idx == lib.EOS_token : 
							tmp_beam_batch[batch_idx].append(beam_batch[batch_idx][beam_idx])
							break

						log_prob = top_values[batch_idx][tmp_beam_idx]
						word_idx = top_idxs[batch_idx][tmp_beam_idx]
						tmp_beam_batch[batch_idx].append(beam.BeamTree(log_prob, word_idx, beam_batch[batch_idx][beam_idx]))

			# get the new beam
			beam_batch = beam.select_topk(beam_size, tmp_beam_batch)
			# check if all nodes in the beam_batch are EOS
			if beam.check_EOS(beam_batch) : break

	top1_batch, seqs_batch = beam.beam_to_seqs(beam_batch)

	# top1_batch(list, [batch] x [seq_length, log_prob]), seqs_batch(list, [batch] x [beam] x [seq_length, log_prob])
	return top1_batch, seqs_batch
