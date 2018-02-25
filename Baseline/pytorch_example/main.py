# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')					# Define the parameters to the CLI
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)																		# Seed the RNG on the GPU if we are using it

###############################################################################
# Load data
###############################################################################
print("Loading data")
corpus = data.Corpus(args.data)																					# Load the data from the disk

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    nbatch = data.size(0) // bsz 																				# Work out how cleanly we can divide the dataset into bsz parts.
    data   = data.narrow(0, 0, nbatch * bsz)																	# Trim off any extra elements that wouldn't cleanly fit (remainders).
    data   = data.view(bsz, -1).t().contiguous()																# Evenly divide the data across the bsz batches.
    if args.cuda:																								# If we can do this on the gpu
        data = data.cuda()																						# Then move the data to the gpu
    return data 																								# Return the answer

print("Batching data")
eval_batch_size = 10																							# Define the batch size for when we want to evaulaute the model
train_data = batchify(corpus.train, args.batch_size) 															# Generate the batches for the training data
val_data   = batchify(corpus.valid, eval_batch_size) 															# Generate the batches for the validation data
test_data  = batchify(corpus.test, eval_batch_size) 															# Generate the batches for the testing data

###############################################################################
# Build the model
###############################################################################

print("Constructing Model")
ntokens = len(corpus.dictionary)																				#Get the size of the vocabulary
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)		#Initialize the model
if args.cuda:																									#If we have access to a GPU
    model.cuda()																								#Load the model on the GPU

criterion = nn.CrossEntropyLoss()																				#Define our loss function as the CrossEntropyLoss

###############################################################################
# Training code
###############################################################################

#Wraps hidden states in new Variables, to detach them from their history.
def repackage_hidden(h):
    if type(h) == Variable:																						# If we are dealing with a variable
        return Variable(h.data)																					# Then extract the data and return it in a new variable
    else:																										# If it isnt a variable, then assume that its iterable
        return tuple(repackage_hidden(v) for v in h)															# Return a tuple containing a recursive call of this function

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    model.eval()																						# Turn on evaluation mode which disables dropout.
    total_loss = 0																						# Define the total loss of the model to be 0 to start
    ntokens = len(corpus.dictionary)																	# Define our vocabulary size
    hidden = model.init_hidden(eval_batch_size)															# Define our hidden states 
    for i in range(0, data_source.size(0) - 1, args.bptt):												# For every batch (batch#, batch starting index)
        data, targets = get_batch(data_source, i, evaluation=True)										# Get the batch in evaulation mode
        output, hidden = model(data, hidden)															# Get the output of the model
        output_flat = output.view(-1, ntokens)															# Get the final output vector from the model (the last word predicted)
        total_loss += len(data) * criterion(output_flat, targets).data                                  # Get the loss of the predicitons
        hidden = repackage_hidden(hidden)																# Reset the hidden states
    return total_loss[0] / len(data_source)																# Return the losses


def train():
    model.train()																						# Turn on training mode which enables dropout.
    total_loss = 0																						# Define the total loss of the model to be 0 to start
    start_time = time.time()																			# Start a timer to keep track of how long our model is taking to train
    ntokens = len(corpus.dictionary)																	# Define our vocabulary size
    hidden = model.init_hidden(args.batch_size)															# Define our hidden states 
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):								# For every batch (batch#, batch starting index)
        data, targets = get_batch(train_data, i)														# Get the batch based on the training data and the batch starting index
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()																				# Before doing our backwards pass make sure that the gradients are all set to zero
        output, hidden = model(data, hidden)															# Based on the current batch, do the forward pass, using the given hidden params
        shaped = output.view(-1, ntokens)
        print shaped.size(), targets.size()
        loss = criterion(shaped, targets)												# Calculate the loss with respect to the last element of the output (we discard all the other outputs here) and the targets
        loss.backward()																					# Actually do the backwards pass, this populates the gradients 

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()																				# The step the optimizer, this actually updates the weights. The optimizer was initialized with the model as a parameter so thats how it keeps track

        total_loss += loss.data 																		# Update the total loss 

        if batch % args.log_interval == 0 and batch > 0:												# If we want to print things out...
            cur_loss = total_loss[0] / args.log_interval												# Calculate the current loss
            elapsed = time.time() - start_time 															# Calculate how much time has passed for this epoch
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr ADAM | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))						# Print some log statement
            total_loss = 0																				# Reset the loss
            start_time = time.time()																	# Reset the time

# Loop over epochs.
lr = args.lr 																							# Define our learning rate
best_val_loss = None																					# Define a best validation loss so we can do early stopping if we want to

optimizer = optim.RMSprop(model.parameters(), lr = args.lr)												# Initialize the optimizer here, pass the parameters from the model so it can update them when its time, also initialize with the passed learning rate
optimizer.zero_grad()																					# Zero the gradient so it can be used from the start

# At any point you can hit Ctrl + C to break out of training early.
try:
    print("Starting training")
    for epoch in range(1, args.epochs+1):																# For every epoch we are going to do
        epoch_start_time = time.time()																	# Set a start time
        train()																							# Train the model
        val_loss = evaluate(val_data)																	# test the model on validation data to check performance
        print('-' * 89)																					
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '							
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))								# Print some log statement
        print('-' * 89)
        
        if not best_val_loss or val_loss < best_val_loss:												# Save the model if the validation loss is the best we've seen so far.
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0																					# Anneal the learning rate if no improvement has been seen in the validation dataset.
except KeyboardInterrupt:																				# If Ctrl + C was pressed
    print('-' * 89)
    print('Exiting from training early')																# Note that it was pressed and continue on

# Load the best saved model.
with open(args.save, 'rb') as f:																		# Load the saved model because training is done and we want the one that performed best on the validation data
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)																			# Evaluate the model on the test data
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))																	# Print the model's performance on the testing data
print('=' * 89)
