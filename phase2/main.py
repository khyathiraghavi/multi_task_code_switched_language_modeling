from constants import *
import argparse
from dataController import data
from models import model
import time
import math
import numpy as np
np.random.seed(28)
import torch
import torch.nn as nn
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--cuda', action='store_false', help='use CUDA')
args = parser.parse_args()

torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(RNG_SEED)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)



class Batching(object):
    def __init__(self):
        self.batching = True

    def batchify(self, data, batch_size, args):
        '''
        identify the number of batches possible with the batch size and then remove the extra data points.
        Distribute the data points equally across the batches
        '''
        num_batches = data.size(0) // batch_size
        data = data.narrow(0, 0, num_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        if args.cuda:
            data = data.cuda()
        return data

    def get_batch(self, source, i, seq_len=None, evaluation=False):
        seq_len = min(seq_len if seq_len else BPTT, len(source) - 1 - i)
        data = Variable(source[i:i+seq_len], volatile=evaluation)
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        return data, target


###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(DATA_PATH)

eval_batch_size = 10
test_batch_size = 1
batching = Batching()
train_data = batching.batchify(corpus.train, BATCH_SIZE, args)
val_data = batching.batchify(corpus.valid, eval_batch_size, args)
test_data = batching.batchify(corpus.test, test_batch_size, args)

print ("done batchifying")

###############################################################################
# Build the model
###############################################################################

num_tokens = len(corpus.dictionary)
model = model.RNNModel(num_tokens)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
#print('Args:', args)
print('Model total parameters:', total_params)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, BPTT):
        data, targets = batching.get_batch(data_source, i, evaluation=True)
        #output, hidden = model(data, hidden)
        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h= True)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    total_loss = 0
    begin_t = time.time()
    #ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(BATCH_SIZE)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = BPTT if np.random.random() < 0.95 else BPTT / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / BPTT
        model.train()
        data, targets = batching.get_batch(train_data, i, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(output.view(-1, num_tokens), targets)

        loss = raw_loss
        # Activiation Regularization
        loss = loss + sum(ALPHA * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(BETA * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), CLIP_GRADIENTS)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % DEUBG_LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss[0] / DEUBG_LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // BPTT, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / DEUBG_LOG_INTERVAL, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = INITIAL_LEARNING_RATE
best_val_loss = []
stored_loss = 100000000


optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for epoch in range(0, EPOCH_MAX):
    epoch_start_time = time.time()
    train()
    if 't0' in optimizer.param_groups[0]:
        tmp = {}
        for prm in model.parameters():
            tmp[prm] = prm.data.clone()
            prm.data = optimizer.state[prm]['ax'].clone()

        val_loss2 = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss2, math.exp(val_loss2)))
        print('-' * 89)

        if val_loss2 < stored_loss:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                torch.save(model, f)
            print('Saving Averaged!')
            stored_loss = val_loss2

        for prm in model.parameters():
            prm.data = tmp[prm].clone()

    else:
        val_loss = evaluate(val_data, eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < stored_loss:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                torch.save(model, f)
            print('Saving Normal!')
            stored_loss = val_loss

        if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>NONMONO and val_loss > min(best_val_loss[:-NONMONO])):
            print('Switching!')
            optimizer = torch.optim.ASGD(model.parameters(), lr=INITIAL_LEARNING_RATE, t0=0, lambd=0., weight_decay=WEIGHT_DECAY)
            #optimizer.param_groups[0]['lr'] /= 2.
        best_val_loss.append(val_loss)


# Load the best saved model.
with open(MODEL_SAVE_PATH, 'rb') as f:
    model = torch.load(f)
    
# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('********************************************************************')
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))

