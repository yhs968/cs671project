from modules.texts import Vocab, GloVeLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import modules.extractive as ext
import modules.abstractive as abs
import modules.beam_search as bs
from modules.data import Documents
from torch.utils.data import DataLoader
import numpy as np

class Test:
    def __init__(self, docs, models):
        assert len(models) == 7
        self.models = models
        self.emb = models[0]
        self.ext_s_enc = models[1]
        self.ext_d_enc = models[2]
        self.ext_extc = models[3]
        self.ext_d_classifier = models[4]
        self.abs_enc = models[5]
        self.abs_dec = models[6]
        self.docs = docs
        from rouge import Rouge
        self.rouge = Rouge()
        
    def run(self, doc_indices, beam_size, verbose=True):
        '''
        Run the test on the specified document indices
        
        Args:
            doc_indices: list of document indices
            beam_size: beam size for the beam search
        '''
        ext_acc, dclass_acc = self.extractive_accuracy(doc_indices)
        r_score = self.rouge1(doc_indices, beam_size, verbose)
        
        print("Sentence Extraction Accuracy: %.3f" % ext_acc)
        print("Document Classification Accuracy: %.3f" % dclass_acc)
        print("Rouge Score: %.3f" % r_score)
        return ext_acc, dclass_acc, r_score
        
    def ext_accuracy(probs, targets):   
        '''
        Calculates the accuracy for the extractor

        Args:
            probs: extraction probability
            targets: ground truth labels for extraction
        '''
        import numpy as np
        preds = np.array([1 if p > 0.5 else 0 for p in probs])
        accuracy = np.mean(preds == targets)

        return accuracy
    
    def extractive_accuracy(self, doc_indices):
    
        ext_s_enc = self.ext_s_enc
        ext_d_enc = self.ext_d_enc
        ext_extc = self.ext_extc
        ext_d_classifier = self.ext_d_classifier
        
        total_accuracy_ext = 0
        total_accuracy_dclass = 0
        n = len(doc_indices)
        
        batch_size = 1

        for i_doc in doc_indices:
            doc = self.docs[i_doc]
            docloader = DataLoader(doc, batch_size=1, shuffle=False)
            # Encode the sentences in a document
            sents_raw = []
            sents_encoded = []
            ext_labels = []
            doc_class = Variable(torch.LongTensor([doc.doc_class])).cuda()
            for sent, ext_label in docloader:
                # only accept sentences that conforms the maximum kernel sizes
                if sent.size(1) < max(self.ext_s_enc.kernel_sizes):
                    continue
                sent = Variable(sent).cuda()
                sents_raw.append(sent)
                sents_encoded.append(ext_s_enc(sent))
                ext_labels.append(ext_label.cuda())
            # Ignore if the content is a single sentence(no need to train)
            if len(sents_raw) <= 1:
                n -= 1
                continue

            # Build the document representation using encoded sentences
            d_encoded = torch.cat(sents_encoded, dim = 0).unsqueeze(1)
            ext_labels = Variable(torch.cat(ext_labels, dim = 0).type(torch.FloatTensor).view(-1)).cuda()
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
            ## Measure the accuracy
            p_cpu = extract_probs.data.cpu().numpy()
            t_cpu = ext_labels.data.cpu().numpy()
            q_cpu = q.data.cpu().numpy()
            c_cpu = doc_class.data.cpu().numpy()

            total_accuracy_ext += Test.ext_accuracy(p_cpu, t_cpu)
            total_accuracy_dclass += int(np.argmax(q_cpu) == c_cpu[0])

        total_accuracy_ext /= n
        total_accuracy_dclass /= n

        return total_accuracy_ext, total_accuracy_dclass

    def rouge1(self, doc_indices, beam_size, verbose):
        '''
        Args:
            beam_size (int)
            doc_indices: list of document indices
            rouge: Rouge() instance
            models: trained models
        '''
        
        rouge = self.rouge
        models = self.models
        rouge_score = 0
        
        n = len(doc_indices)

        for i_doc in doc_indices:
            try:
                test_input = [torch.LongTensor(sent).view(1,-1) for sent in self.docs[i_doc].sents]
                ref_input = torch.LongTensor(self.docs[i_doc].head).view(1,-1)
                top1_batch, seqs_batch = bs.generate_title(doc_sents = test_input,
                                                           beam_size = 5,
                                                           models = models,
                                                           max_kernel_size = max(self.ext_s_enc.kernel_sizes))
                rouge_batch = 0
                for i in range(len(top1_batch)):
                    generated = '_BEGIN_ ' + self.docs.vocab.id2sents([top1_batch[i][0]])
                    reference = self.docs.vocab.id2sents([ref_input[i]])
                    rouge_batch += rouge.get_scores(generated, reference)[0]['rouge-1']['f']
                rouge_batch /= len(top1_batch)
                rouge_score += rouge_batch
            except Exception as e:
                if verbose:
                    print(e)
                n -= 1

        rouge_score /= n

        return rouge_score