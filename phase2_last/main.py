from constants import *
import argparse
from dataController import data
from dataController.Batching import *
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
    print("Using cuda")
    model.cuda()


# define loss
criterion = nn.CrossEntropyLoss()
loss_least = 100000000


def evaluate(data, mode):
    if mode == 'val_mode':
        bsz = val_batch_size
    elif mode == 'test_mode':
        bsz = test_batch_size
    global loss_least
    global num_tokens
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
    print("Epoch: "+str(epoch)+" Val Loss: " + str(final_loss) + " Val Perplexity: " + str(math.exp(final_loss)))
    '''
    if final_loss < loss_least:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                torch.save(model, f)
            loss_least = final_loss
    '''
    return final_loss


optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
stochastic = True

def saving_model(final_loss):
    global loss_least
    if final_loss < loss_least:
            with open(MODEL_SAVE_PATH, 'wb') as f:
                torch.save(model, f)
            loss_least = final_loss
    return

def train(stochastic):
    model.train()
    global optimizer
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

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * window / BPTT

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

# Loop over epochs.
lr = INITIAL_LEARNING_RATE
val_loss_interval = []


for epoch in range(0, EPOCH_MAX):
    epoch_start_time = time.time()
    train(stochastic)
    if stochastic == False: #in asgd

        loss_val = evaluate(val_data, 'val_mode')

        param_temp = {}
        for prm in model.parameters():
            param_temp[prm] = prm.data.clone()
            prm.data = optimizer.state[prm]['ax'].clone()
        
        saving_model(loss_val)
        for prm in model.parameters():
            prm.data = param_temp[prm].clone()

    else: # sgd
        val_loss = evaluate(val_data, 'val_mode')

        if stochastic == True:
            t = len(val_loss_interval)
            if (t>NONMONO and val_loss > min(val_loss_interval[:-NONMONO])):
                print('Triggering ASGD')
                stochastic = False
        val_loss_interval.append(val_loss)


# Loading the best model
with open(MODEL_SAVE_PATH, 'rb') as f:
    model = torch.load(f)
    
# Testing
final_loss = evaluate(test_data, 'test_mode')
print("Test Loss: " + str(final_loss) + " test Perplexity: " + str(math.exp(final_loss)))
