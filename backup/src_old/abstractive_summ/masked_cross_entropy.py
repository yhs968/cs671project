import torch
from torch.nn import functional
from torch.autograd import Variable


def sequence_mask(seq_lengths):
	max_len = seq_lengths.data.max()
	batch_size = seq_lengths.size(0)
	seq_range = torch.arange(0, max_len).long()
	seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
	seq_range_expand = Variable(seq_range_expand).cuda()
	seq_lengths_expand = (seq_lengths.unsqueeze(1).expand_as(seq_range_expand))
	return seq_range_expand < seq_lengths_expand


def masked_cross_entropy(logits, target, lengths):
	logits_flat = logits.view(-1, logits.size(-1)) # batch * max_len x num_classes
	log_probs_flat = functional.log_softmax(logits_flat) # batch * max_len x num_classes
	target_flat = target.view(-1, 1) # batch * max_len x 1
	losses_flat = -torch.gather(log_probs_flat, dim = 1, index = target_flat) # batch * max_len x 1
	losses = losses_flat.view(target.size()) # batch x max_len
	lengths = Variable(torch.LongTensor(lengths)).cuda()
	mask = sequence_mask(lengths) # mask : batch x max_len
	losses = losses * mask.float()
	loss = losses.sum() / lengths.float().sum()
	return loss
