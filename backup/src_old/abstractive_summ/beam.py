import lib

# select topk(beam_size) nodes for each batch
def select_topk(beam_size, tmp_beam_batch) : 
    beam_batch = []
    for tmp_beam in tmp_beam_batch :
        beam = []
        # firstly select previously EOS node
        for node in tmp_beam : 
            if node.is_done : beam.append(node)
        
        # remove selected node from tmp_beam_batch
        for node in beam : tmp_beam.remove(node)	

        # sort by total log probability
        sorted_tmp_beam = sorted(tmp_beam, key = lambda BeamTree : BeamTree.total_log_prob, reverse = True)
        for beam_idx in range(beam_size - len(beam)) : 
            beam.append(sorted_tmp_beam[beam_idx])
            if sorted_tmp_beam[beam_idx].word_idx == lib.EOS_token : sorted_tmp_beam[beam_idx].is_done = True
        beam_batch.append(beam)
    
    return beam_batch

# check if all nodes in the beam_batch are EOS
def check_EOS(beam_batch) : 
    for beam in beam_batch : 
        for node in beam : 
            if not node.is_done : return False 
            
    return True

# get seqs from beam_batch
def beam_to_seqs(beam_batch) : 
    seqs_batch = []
    top1_batch = []
    for beam in beam_batch : 
        # sort each beam
        sorted_beam = sorted(beam, key = lambda BeamTree : BeamTree.total_log_prob, reverse = True)
        # get str from node	
        seqs = []
        for node in sorted_beam : 
            seqs.append([node_to_seq(node), node.total_log_prob])
        seqs_batch.append(seqs)
        top1_batch.append(seqs[0])

    return top1_batch, seqs_batch

# get seq from leaf node
def node_to_seq(leaf) : 
    seq = []
    cur = leaf
    while True : 
        seq.append(cur.word_idx)
        cur = cur.parent_node
        if cur.word_idx == lib.SOS_token : 
            break

    seq.reverse()
    return seq

# just for easy printing
def print_beam_batch(beam_batch) : 
    for beam in beam_batch : 
        print(beam)
    print()

class BeamTree : 
    def __init__(self, log_prob, word_idx, parent_node = None) : 
        self.log_prob = log_prob
        self.word_idx = word_idx
        self.parent_node = parent_node
        # log probs from the root (all log_probs added)
        if parent_node is None : self.total_log_prob = 0
        else : self.total_log_prob = parent_node.log_prob + log_prob
        # whether this node is EOS
        self.is_done = False

    def __repr__(self) : 
        return '[%d, %.2f, %.2f, %s]' % (self.word_idx, self.log_prob, self.total_log_prob, self.is_done)
