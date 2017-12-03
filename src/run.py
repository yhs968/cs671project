import sys, getopt

def main(argv):
    hidden_size = 400
    n_kernels = 50
    try:
        opts, args = getopt.getopt(argv, "hkst:o:", ["h=","k=","s=","t="])
    except getopt.GetoptError:
        print('run.py --h <hidden_size> --k <n_kernels> --s <n_samples> --t <n_tests>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--h"):
            hidden_size = int(arg)
        elif opt in ("-k", "--k"):
            n_kernels = int(arg)
        elif opt in ("-s", "--s"):
            n_samples = int(arg)

    print('hidden_size: %3i' % hidden_size)
    print('n_kernels: %3i' % n_kernels)
    print('n_samples: %3i' % n_samples)
    
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
    from modules.test import Test

    import numpy as np

    # Load the pretrained embedding into the memory
    path_glove = os.path.join(os.path.expanduser('~'),
                 'data/NLP/word_embeddings/GloVe/glove.6B.200d.txt')
    glove = GloVeLoader(path_glove)

    # Load the dataset
    doc_file = './data/kaggle_news_rouge1.pkl'
    docs = Documents(doc_file, vocab_size = 30000)
    vocab = docs.vocab

    d = 200
    emb = nn.Embedding(vocab.V, d)

    def init_emb(emb, vocab):
        for word in vocab.word2id:
            try:
                emb.weight.data[vocab[word]] = torch.from_numpy(glove[word])
            except KeyError as e:
                # Case when pretrained embedding for a word does not exist
                pass
    #     emb.weight.requires_grad = False # suppress updates
        print('Initialized the word embeddings.')

    # Test
    from copy import deepcopy
    from torch import optim
    import time
    from itertools import chain

    vocab_size = vocab.V
    emb_size = emb.weight.data.size(1)
    n_kernels = n_kernels
    kernel_sizes = [1,2,3,4,5]
    pretrained = emb
    sent_size = len(kernel_sizes) * n_kernels
    hidden_size = hidden_size
    num_layers = 1
    n_classes = len(docs.dclass2id)
    batch_size = 1
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    init_emb(emb, vocab)
    ext_s_enc = ext.SentenceEncoder(vocab_size, emb_size,
                                       n_kernels, kernel_sizes, pretrained)
    ext_d_enc = ext.DocumentEncoder(sent_size, hidden_size)
    ext_extc = ext.ExtractorCell(sent_size, hidden_size)
    ext_d_classifier = ext.DocumentClassifier(sent_size, n_classes)
    abs_enc = abs.EncoderRNN(emb, hidden_size, num_layers)
    abs_dec = abs.AttnDecoderRNN(emb, hidden_size * 2, num_layers)

    models = [emb, ext_s_enc, ext_d_enc, ext_extc, ext_d_classifier,
             abs_enc, abs_dec]
    params = list(chain(*[model.parameters() for model in models]))
    optimizer = optim.SGD(params, lr = .005)

    loss_fn_ext = nn.BCELoss()
    loss_fn_dclass = nn.NLLLoss()
    loss_fn_abs = nn.CrossEntropyLoss()

    def get_accuracy(probs, targets, verbose = False):   
        '''
        Calculates the accuracy for the extractor

        Args:
            probs: extraction probability
            targets: ground truth labels for extraction
        '''
        import numpy as np
        preds = np.array([1 if p > 0.5 else 0 for p in probs])
        if verbose:
            print(preds)
        accuracy = np.mean(preds == targets)

        return accuracy

    # class RougeScorer:
    #     def __init__(self):
    #         from rouge import Rouge
    #         self.rouge = Rouge()
    #     def score(self, reference, generated, type = 1):
    #         score = self.rouge.get_scores(reference, generated, avg=True)
    #         score = score['rouge-%s' % type]['f']
    #         return score

    # rouge = RougeScorer()

    def run_epoch(docs, n_samples = 3000):

        epoch_loss_abs = 0
        epoch_loss_ext = 0
        epoch_loss_dclass = 0
        epoch_accuracy_ext = 0
        epoch_accuracy_dclass = 0
        n = n_samples

        for i, doc in enumerate(docs):
            if i >= n_samples: break
            optimizer.zero_grad()
            docloader = DataLoader(doc, batch_size=1, shuffle=False)
            # Encode the sentences in a document
            sents_raw = []
            sents_encoded = []
            ext_labels = []
            doc_class = Variable(torch.LongTensor([doc.doc_class])).cuda()
            for sent, ext_label in docloader:
                # only accept sentences that conforms the maximum kernel sizes
                if sent.size(1) < max(kernel_sizes):
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

            ## Optimize over the extractive examples
            loss_ext = loss_fn_ext(extract_probs, ext_labels)
            loss_dclass = loss_fn_dclass(q.view(1,-1), doc_class)
            epoch_loss_ext += loss_ext.data.cpu().numpy()[0]
            epoch_loss_dclass += loss_dclass.data.cpu().numpy()[0]
            torch.autograd.backward([loss_ext, loss_dclass])
            optimizer.step()

            ## Measure the accuracy
            p_cpu = extract_probs.data.cpu().numpy()
            t_cpu = ext_labels.data.cpu().numpy()
            q_cpu = q.data.cpu().numpy()
            c_cpu = doc_class.data.cpu().numpy()
            epoch_accuracy_ext += get_accuracy(p_cpu, t_cpu)
            epoch_accuracy_dclass += int(np.argmax(q_cpu) == c_cpu[0])

            # Abstractive Summarizer
            optimizer.zero_grad()
            loss_abs = 0
            ## Run through the encoder
    #         words = torch.cat(sents_ext, dim=1).t()
            sents_ext = [sent for i,sent in enumerate(sents_raw)
                         if extract_probs[i].data.cpu().numpy() > 0.5]

            # skip if no sentences are selected as summaries
            if len(sents_ext) == 0:
                n -= 1
                continue
            words = torch.cat(sents_ext, dim=1).t()

            abs_enc_hidden = abs_enc.init_hidden(batch_size)
            abs_enc_output, abs_enc_hidden = abs_enc(words, abs_enc_hidden)
            ## Remove to too long documents to tackle memory overflow
            if len(abs_enc_output) > 6000:
                n -=1
                continue
            ## Run through the decoder
            abs_dec_hidden = abs_enc_hidden.view(1,1,-1)
    #         abs_dec_hidden = abs_enc_hidden
            for i in range(len(doc.head)-1):
                input = doc.head[i]
                target = doc.head[i+1]
                input = Variable(torch.LongTensor([input]).unsqueeze(1)).cuda()
                target = Variable(torch.LongTensor([target]).unsqueeze(1)).cuda()
                abs_dec_output, abs_dec_hidden, _ = abs_dec(input, abs_dec_hidden, abs_enc_output)
                loss_abs += loss_fn_abs(abs_dec_output, target.squeeze(1))

            epoch_loss_abs += loss_abs.data.cpu().numpy()[0]
            loss_abs.backward()
            optimizer.step()

        epoch_accuracy_ext /= n
        epoch_accuracy_dclass /= n

        return epoch_loss_ext, epoch_loss_dclass, epoch_loss_abs, epoch_accuracy_ext, epoch_accuracy_dclass

    def train(docs, n_epochs = 10, n_samples = 3000, print_every = 1):
        import time

        start_time = time.time()
        for epoch in range(n_epochs):
            ext_loss, dclass_loss, abs_loss, ext_acc, dclass_acc = run_epoch(docs, n_samples)
            if epoch % print_every == 0:
                end_time = time.time()
                wall_clock = (end_time - start_time) / 60
                print('Epoch:%2i / Loss:(%.3f/%.3f/%.3f) / Accuracy:(%.3f/%.3f) / TrainingTime:%.3f(min)' %
                      (epoch, ext_loss, dclass_loss, abs_loss, ext_acc, dclass_acc, wall_clock))
                start_time = time.time()

    import os
    from os.path import join            
    # Training
    # train(docs, n_epochs = 50, n_samples = 10, print_every = 10)
    save_every = 5
    for n in range(20):
        train(docs, n_epochs = save_every, n_samples = n_samples, print_every = 1)
        print('Epoch %2i finished.' % ((n+1)*save_every))
        model_dict = dict()
        model_dict['emb'] = emb
        model_dict['ext_s_enc'] = ext_s_enc
        model_dict['ext_d_enc'] = ext_d_enc
        model_dict['ext_extc'] = ext_extc
        model_dict['ext_d_classifier'] = ext_d_classifier
        model_dict['abs_enc'] = abs_enc
        model_dict['abs_dec'] = abs_dec

        data_dir = join(os.path.expanduser('~'), 'cs671-large')
        for name, model in model_dict.items():
            torch.save(model.state_dict(), join(data_dir, 'hidden%inkernels%iepoch%i_' % (hidden_size, n_kernels, (n+1)*save_every)) + name) 
        # Testing
        tester = Test(docs, models)
        ext_acc, dclass_acc, r_score = tester.run(range(n_samples, len(docs)), 20)

        with open('./data/results%s.txt' % ('_hidden%inkernels%i' % (hidden_size, n_kernels)), 'w') as f:
            s0 = '[hidden_size: %3i, n_kernels: %3i]' % (hidden_size, n_kernels)
            s1 = "Sentence Extraction Accuracy: %.3f" % ext_acc
            s2 = "Document Classification Accuracy: %.3f" % dclass_acc
            s3 = "Rouge Score: %.3f" % r_score
            f.write('\n'.join([s1,s2,s3]))

if __name__ == "__main__":
    main(sys.argv[1:])
