from rouge import Rouge

import lib
import generate

def evaluate(input_batch, input_lengths, target_batch, max_target_length, beam_size, encoder, decoder, output_lang, n_print_per_batch, rouge) : 
	top1_batch, seqs_batch = generate.generate(input_batch, input_lengths, max_target_length, beam_size, encoder, decoder, output_lang)

	target_batch = target_batch.transpose(0, 1).data
	total_rouge_score = 0
	for idx in range(len(top1_batch)) :
		generated = lib.idxs_to_sentence(output_lang, top1_batch[idx][0])
		target_sentence = lib.idxs_to_sentence(output_lang, target_batch[idx])
		score = rouge.get_scores(generated, target_sentence)[0]['rouge-1']['f']
		total_rouge_score += score

		if idx < n_print_per_batch : 
			print('generated : ', generated)
			print('answer    : ', target_sentence)
			print()

	return total_rouge_score / len(top1_batch)
