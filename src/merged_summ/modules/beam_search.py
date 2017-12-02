import torch
import torch.nn as nn
from torch.autograd import Variable
import modules.texts as texts
import torch.nn.functional as F

class BeamTree : 
    def __init__(self, log_prob, word_idx, parent_node = None) : 
        self.log_prob = log_prob
        self.word_idx = word_idx
        self.parent_node = parent_node
        # log probs from the root (all log_probs added)
        if parent_node is None : self.total_log_prob = 0
        else: self.total_log_prob = parent_node.log_prob + log_prob
        # whether this node is EOS
        self.is_done = False

    def __repr__(self) : 
        return '[%d, %.2f, %.2f, %s]' % (self.word_idx, self.log_prob, self.total_log_prob, self.is_done)
    
# select topk(beam_size) nodes for each batch
def select_topk(beam_size, tmp_beam_batch):
    beam_batch = []
    for tmp_beam in tmp_beam_batch:
        beam = []
        # firstly select previously EOS node
        for node in tmp_beam: 
            if node.is_done: beam.append(node)
        
        # remove selected node from tmp_beam_batch
        for node in beam: tmp_beam.remove(node)

        # sort by total log probability
        sorted_tmp_beam = sorted(tmp_beam, key = lambda BeamTree : -BeamTree.total_log_prob)
        for beam_idx in range(beam_size - len(beam)) : 
            beam.append(sorted_tmp_beam[beam_idx])
            if sorted_tmp_beam[beam_idx].word_idx == texts.EOS_token : sorted_tmp_beam[beam_idx].is_done = True
        beam_batch.append(beam)
    
    return beam_batch

def check_EOS(beam_batch):
    '''
    Checks if all nodes in the beam_batch are EOS
    '''
    for beam in beam_batch:
        for node in beam:
            if not node.is_done: return False
    
    return True

def beam2seq(beam_batch):
    seqs_batch = []
    top1_batch = []
    for beam in beam_batch:
        # sort each beam
        sorted_beam = sorted(beam, key = lambda BeamTree: -BeamTree.total_log_prob)
        # get strings from node
        seqs = []
        for node in sorted_beam:
            seqs.append([node2seq(node), node.total_log_prob])
        seqs_batch.append(seqs)
        top1_batch.append(seqs[0])
        
    return top1_batch, seqs_batch

def node2seq(leaf):
    '''
    Get a 
    '''
    seq = []
    cur = leaf
    while True:
        seq.append(cur.word_idx)
        cur = cur.parent_node
        if cur.word_idx == texts.SOS_token: break
    seq.reverse()
    
    return seq

def generate_title(doc_sents, beam_size, max_kernel_size, models, max_target_length = 100, batch_size = 1):
    '''
    Args:
        doc_sents: list of torch.LongTensors, where each elements can
        have variable length.
        beam_size (int)
        models (list): encoders and decoders
        max_kernel_size (int): maximum kernel size of the CNN sentence encoder
        max_target_length (int): maximum length that a sentence can have.
    '''
    assert len(models) == 6
    
    ext_s_enc = models[0]
    ext_d_enc = models[1]
    ext_extc = models[2]
    ext_d_classifier = models[3]
    abs_enc = models[4]
    abs_dec = models[5]
    
    # Encode the sentences in a document
    if len(doc_sents) <= 1:
        print('Error: The length of the document is %i.' % 1)
        return
    
    sents_raw = []
    sents_encoded = []
    for sent in doc_sents:
        if sent.size(1) < max_kernel_size:
            continue
        sent = Variable(sent).cuda()
        sents_raw.append(sent)
        sents_encoded.append(ext_s_enc(sent))
    
    # Build the document representation using encoded sentences
    d_encoded = torch.cat(sents_encoded, dim = 0).unsqueeze(1)
    init_sent = ext_s_enc.init_sent(batch_size)
    d_ext = torch.cat([init_sent, d_encoded[:-1]], dim = 0)
    
    # Extractive Summarizer
    ## Initialize the d_encoder
    h, c = ext_d_enc.init_h0c0(batch_size)
    h0 = Variable(h.data)
    ## An input goes through the document encoder
    output, hn, cn = ext_d_enc(d_ext, h, c)
    ## Initialize the decoder
    ### calculate p0, h_bar0, c_bar0
    h_ = hn.squeeze(0)
    c_ = cn.squeeze(0)
    p = ext_extc.init_p(h0.squeeze(0), h_)
    ### calculate p_t, h_bar_t, c_bar_t
    d_encoder_hiddens = torch.cat((h0, output[:-1]), 0) #h0 ~ h_{n-1}
    extract_probs = Variable(torch.zeros(len(sents_encoded))).cuda()
    for i, (s, h) in enumerate(zip(sents_encoded, d_encoder_hiddens)):
        h_, c_, p = ext_extc(s, h, h_, c_, p)
        extract_probs[i] = p.squeeze(0)
    ## Document Classifier
    q = ext_d_classifier(extract_probs.view(-1,1), d_encoded.squeeze(1))
    
    # Abstractive Summarizer
    sents_ext = [sent for i,sent in enumerate(sents_raw)
                 if extract_probs[i].data[0] > 0.5]
    ## skip if no sentences are selected as summaries
    if len(sents_ext) == 0:
        print("No sentences are selected")
        return
    words = torch.cat(sents_ext, dim=1).t()
    abs_enc_hidden = abs_enc.init_hidden(batch_size)
    abs_enc_output, abs_enc_hidden = abs_enc(words, abs_enc_hidden)
    ## Remove to too long documents to tackle memory overflow
    if len(abs_enc_output) > 6000:
        print('Out of memory')
        return
    abs_dec_hidden = abs_dec.init_hidden(batch_size)
    abs_dec_input = Variable(torch.LongTensor([texts.SOS_token]).unsqueeze(1)).cuda()
    
    beam_batch = [[] for i in range(batch_size)]
    
    for t in range(max_target_length):
        if t == 0:
            abs_dec_output, abs_dec_hidden, _ = abs_dec(abs_dec_input, abs_dec_hidden, abs_enc_output)
            # (B = 1, V = vocab_size)
            abs_dec_prob = F.log_softmax(abs_dec_output)
            # print(abs_dec_prob.size())
            # Get top-k(beam size) values
            top_values, top_idxs = abs_dec_prob.data.topk(beam_size, dim = -1)
            # print(top_values.size())
            for batch_idx in range(batch_size):
                log_prob = 0 # p = 1
                root = BeamTree(log_prob, texts.SOS_token)
                for beam_idx in range(beam_size):
                    log_prob = top_values[batch_idx][beam_idx]
                    word_idx = top_idxs[batch_idx][beam_idx]
                    beam_batch[batch_idx].append(BeamTree(log_prob, word_idx, root))
        else:
            tmp_beam_batch = [[] for i in range(batch_size)]
            for beam_idx in range(beam_size): 
                abs_dec_input = []
                for batch_idx in range(batch_size): 
                    # decoder inputs are words in current beam
                    abs_dec_input.append(beam_batch[batch_idx][beam_idx].word_idx)
                # Regard each beams as seperate batches
                abs_dec_input = Variable(torch.LongTensor(abs_dec_input).view(1,-1)).cuda()
                abs_dec_output, abs_dec_hidden, attn_weights = abs_dec(abs_dec_input, abs_dec_hidden, abs_enc_output)

                # get top k(beam size) values
                top_values, top_idxs = abs_dec_prob.data.topk(beam_size, dim = -1)
                for batch_idx in range(batch_size) :
                    for tmp_beam_idx in range(beam_size) : 
                        # if current word is EOS, add it to tmp_beam_batch instead of its children
                        if beam_batch[batch_idx][beam_idx].word_idx == texts.EOS_token: 
                            tmp_beam_batch[batch_idx].append(beam_batch[batch_idx][beam_idx])
                            break
                        log_prob = top_values[batch_idx][tmp_beam_idx]
                        word_idx = top_idxs[batch_idx][tmp_beam_idx]
                        tmp_beam_batch[batch_idx].append(BeamTree(log_prob, word_idx, beam_batch[batch_idx][beam_idx]))

            # get the new beam
            beam_batch = select_topk(beam_size, tmp_beam_batch)
            # check if all nodes in the beam_batch are EOS
            if check_EOS(beam_batch): break
    
    top1_batch, seqs_batch = beam2seq(beam_batch)
    
    return top1_batch, seqs_batch