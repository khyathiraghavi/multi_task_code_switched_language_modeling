from constants import *
import argparse
from dataController import data
from models import model
import time
import math
import numpy as np
np.random.seed(331)
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


##### LOADING DATA #####

corpus = data.Corpus(DATA_PATH)
# retaining the same batch size to reproduce the results.
val_batch_size = 10
test_batch_size = 1
batching = Batching()
train_data = batching.batchify(corpus.train, BATCH_SIZE, args)
val_data = batching.batchify(corpus.valid, val_batch_size, args)
test_data = batching.batchify(corpus.test, test_batch_size, args)

print ("done batchifying")



num_tokens = len(corpus.dictionary)
model = model.RNNModel(num_tokens)

# enable GPU model when run with cuda
if args.cuda:
    print('using cuda')
    model.cuda()


# define loss
criterion = nn.CrossEntropyLoss()


def evaluate(data, mode):
    if mode == 'val_mode':
        bsz = val_batch_size
    elif mode == 'test_mode':
        bsz = test_batch_size
    global loss_least
    global num_tokens
    if MODEL_TYPE == "QRNN":
        model.reset()
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(bsz)
    for i in range(0, data.size(0) - 1, BPTT):
        X, Y = batching.get_batch(data, i, evaluation=True)
        output, hidden = model(X, hidden)
        predictions = output.view(-1, num_tokens)
        total_loss += len(X) * criterion(predictions, Y).data
        hidden = repackage_hidden(hidden)
    final_loss = total_loss[0] / len(data)
    try:
        print("Epoch: "+str(epoch)+" Val Loss: " + str(final_loss) + " Val Perplexity: " + str(math.exp(final_loss)))
    except:
        print("Val Loss: " + str(final_loss) + " Val Perplexity: " + str(math.exp(final_loss)))

    
    if final_loss < loss_least:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                torch.save(model, f)
            loss_least = final_loss
    return final_loss
    

optimizer = torch.optim.ASGD(model.parameters(), lr=INITIAL_LEARNING_RATE, t0=0, lambd=0, weight_decay=WEIGHT_DECAY)
stochastic = False


def saving_model(final_loss):
    global loss_least
    if final_loss < loss_least:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                torch.save(model, f)
            loss_least = final_loss
    return


def train(stochastic):
    global optimizer
    if MODEL_TYPE == "QRNN":
        model.reset()
    if stochastic == False:
        optimizer = torch.optim.ASGD(model.parameters(), lr=INITIAL_LEARNING_RATE, t0=0, \
            lambd=0., weight_decay=WEIGHT_DECAY)
    
    total_loss = 0
    begin_t = time.time()
    hidden = model.init_hidden(BATCH_SIZE)
    i = 0
    while i < train_data.size(0)-2:
        prob = 0.95
        rand_prob = np.random.random()
        if rand_prob < prob:
            bptt = BPTT
        else:
            bptt = BPTT/2
        s = 5
        window = max(s, int(np.random.normal(bptt, s)))
        window = min(window, BPTT+10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * window / BPTT

        model.train()
        X, Y = batching.get_batch(train_data, i, seq_len=window)

        hidden = repackage_hidden(hidden)
        optimizer.zero_grad() # NOT SURE

        output, hidden = model(X, hidden)
        loss_base = criterion(output.view(-1, num_tokens), Y)
        ar_loss =  ALPHA * model.ar_fragment
        tar_loss = BETA * model.tar_fragment
        
        loss = loss_base + ar_loss + tar_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), CLIP_GRADIENTS)

        optimizer.step()

        total_loss += loss_base.data
        optimizer.param_groups[0]['lr'] = lr2
        i += window

with open(MODEL_SAVE_PATH, 'rb') as m:
    model = torch.load(m)

# Loop over epochs.
lr = INITIAL_LEARNING_RATE
val_loss_interval = []

loss_least = evaluate(val_data, 'val_mode')


for epoch in range(0, EPOCH_MAX):
    epoch_start_time = time.time()
    train(stochastic)
    if stochastic == False: #in asgd

        loss_val = evaluate(val_data, 'val_mode')

        tmp = {}
        for prm in model.parameters():
            tmp[prm] = prm.data.clone()
            prm.data = optimizer.state[prm]['ax'].clone()

        saving_model(loss_val)
        
        for prm in model.parameters():
            prm.data = tmp[prm].clone()

    #else: # sgd
    #val_loss = evaluate(val_data, 'val_mode')

    #if stochastic == True:
    t = len(val_loss_interval)
    if (t>NONMONO and val_loss > min(val_loss_interval[:-NONMONO])):
        print('Triggering ASGD')
        
    val_loss_interval.append(val_loss)


# Loading the best model
with open(MODEL_SAVE_PATH, 'rb') as f:
    model = torch.load(f)
    
# Testing
final_loss = evaluate(test_data, 'test_mode')
print("Test Loss: " + str(final_loss) + " test Perplexity: " + str(math.exp(final_loss)))
