from constants import MODEL_TYPE, DATA_PATH, MODEL_SAVE_PATH, WORD_FACTOR, LANGUAGE_FACTOR, SOUNDEX_FACTOR, FACTORS_LIST, WORD_EMBEDDING_SIZE, LANG_EMBEDDING_SIZE, SOUNDEX_EMBEDDING_SIZE, NUMBER_HIDDEN, NUMBER_LAYERS, INITIAL_LEARNING_RATE, CLIP_GRADIENTS, EPOCH_MAX,  EVAL_BATCH_SIZE, BATCH_SIZE, MAX_SEQ_LEN, DROPOUT_PERCENT,  RNG_SEED, USE_CUDA, DEUBG_LOG_INTERVAL
from dataController.FactorsInfo import FactorsInfo
from dataController.data import Corpus
from models import model

import numpy as np
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

def batchify(data, bsz, factorCount):
	nbatch = data.size(0) // bsz 																				# Work out how cleanly we can divide the dataset into bsz parts.
	data   = data.narrow(0, 0, nbatch * bsz)																	# Trim off any extra elements that wouldn't cleanly fit (remainders).
	data   = data.view(bsz, -1, factorCount).contiguous()																# Evenly divide the data across the bsz batches.
	if USE_CUDA:																								# If we can do this on the gpu
		data = data.cuda()																						# Then move the data to the gpu
	return data 																								# Return the answer

# Set the random seed manually for reproducibility.
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
	if not USE_CUDA:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		torch.cuda.manual_seed(RNG_SEED)																		# Seed the RNG on the GPU if we are using it

###############################################################################
# Load data
###############################################################################
print("Loading data")
corpus = Corpus(DATA_PATH, FACTORS_LIST)																	# Load the data from the disk

myFactorsInfo = FactorsInfo()
myFactorsInfo.setFactorDetails(WORD_FACTOR, len(corpus.getFactorDict(WORD_FACTOR)), WORD_EMBEDDING_SIZE)
myFactorsInfo.setFactorDetails(LANGUAGE_FACTOR, len(corpus.getFactorDict(LANGUAGE_FACTOR)), LANG_EMBEDDING_SIZE)
myFactorsInfo.setFactorDetails(SOUNDEX_FACTOR, len(corpus.getFactorDict(SOUNDEX_FACTOR)), SOUNDEX_EMBEDDING_SIZE)

trainingData   = corpus.convertToData(DATA_PATH+".train", myFactorsInfo)
validationData = corpus.convertToData(DATA_PATH+".val", myFactorsInfo)
testingData    = corpus.convertToData(DATA_PATH+".test", myFactorsInfo)
print("data loaded and factors extracted")
print("Training data", trainingData.size())
print("Validation data", validationData.size())
print("Testing data", testingData.size())

print("Batching data")
train_data = batchify(trainingData,   BATCH_SIZE,      len(FACTORS_LIST)) 								    						# Generate the batches for the training data
val_data   = batchify(validationData, EVAL_BATCH_SIZE, len(FACTORS_LIST))  															# Generate the batches for the validation data
test_data  = batchify(testingData,    EVAL_BATCH_SIZE, len(FACTORS_LIST))  															# Generate the batches for the testing data

###############################################################################
# Build the model
###############################################################################

print("Constructing Model")
outputVocabSize = len(corpus.getFactorDict(WORD_FACTOR))
model = model.FactoredRNN(MODEL_TYPE, myFactorsInfo, outputVocabSize, NUMBER_HIDDEN, NUMBER_LAYERS, DROPOUT_PERCENT)		#Initialize the model
if USE_CUDA:																												#If we have access to a GPU
	model.cuda()																											#Load the model on the GPU

criterion = nn.CrossEntropyLoss()																							#Define our loss function as the CrossEntropyLoss

###############################################################################
# Training code
###############################################################################

#Wraps hidden states in new Variables, to detach them from their history.
def repackage_hidden(h):
	if type(h) == Variable:																						# If we are dealing with a variable
		return Variable(h.data)																					# Then extract the data and return it in a new variable
	else:																										# If it isnt a variable, then assume that its iterable
		return tuple(repackage_hidden(v) for v in h)															# Return a tuple containing a recursive call of this function

def get_batch(source, i, fInfo, evaluation=False):
	seq_len = min(MAX_SEQ_LEN, source.size()[1] - 1 - i)
	data = Variable(source.narrow(1, i, seq_len), volatile=evaluation)
	target = source.narrow(1, i+1, seq_len)
	start = fInfo.getFactorOrder().index(WORD_FACTOR)
	target = target.narrow(2, start, 1)
	target = target.contiguous().view(-1)
	target = Variable(target)
	return data, target


def evaluate(data_source):
	model.eval()																						# Turn on evaluation mode which disables dropout.
	total_loss = 0																						# Define the total loss of the model to be 0 to start
	ntokens = myFactorsInfo.getFactorVocabSize(WORD_FACTOR)																		# Define our vocabulary size
	hidden = model.init_hidden(EVAL_BATCH_SIZE)															# Define our hidden states 
	for i in range(0, data_source.size(0) - 1, MAX_SEQ_LEN):											# For every batch (batch#, batch starting index)
		data, targets = get_batch(data_source, i, myFactorsInfo, evaluation=True)										# Get the batch in evaulation mode
		output, hidden = model(data, hidden)															# Get the output of the model
		output_flat = output.view(-1, ntokens)															# Get the final output vector from the model (the last word predicted)
		total_loss += len(data) * criterion(output_flat, targets).data                                  # Get the loss of the predicitons
		hidden = repackage_hidden(hidden)																# Reset the hidden states
	return total_loss[0] / len(data_source)																# Return the losses


def train():
	model.train()																						# Turn on training mode which enables dropout.
	total_loss = 0																						# Define the total loss of the model to be 0 to start
	start_time = time.time()																			# Start a timer to keep track of how long our model is taking to train
	ntokens = myFactorsInfo.getFactorVocabSize(WORD_FACTOR)																	# Define our vocabulary size
	hidden = model.init_hidden(BATCH_SIZE)														        # Define our hidden states 
	trainOrder = range(0, train_data.size()[1] - 1, MAX_SEQ_LEN)
	np.random.shuffle(trainOrder)
	for batch, i in enumerate(trainOrder):							# For every batch (batch#, batch starting index)
		data, targets = get_batch(train_data, i, myFactorsInfo)														# Get the batch based on the training data and the batch starting index
		# Starting each batch, we detach the hidden state from how it was previously produced.
		# If we didn't, the model would try backpropagating all the way to start of the dataset.
		hidden = repackage_hidden(hidden)
		model.zero_grad()																				# Before doing our backwards pass make sure that the gradients are all set to zero
		output, hidden = model(data, hidden)															# Based on the current batch, do the forward pass, using the given hidden params
		loss = criterion(output.view(-1, ntokens), targets)												# Calculate the loss with respect to the last element of the output (we discard all the other outputs here) and the targets
		loss.backward()																					# Actually do the backwards pass, this populates the gradients 

		# `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
		torch.nn.utils.clip_grad_norm(model.parameters(), CLIP_GRADIENTS)
		optimizer.step()																				# The step the optimizer, this actually updates the weights. The optimizer was initialized with the model as a parameter so thats how it keeps track

		total_loss += loss.data 																		# Update the total loss 

		if batch % DEUBG_LOG_INTERVAL == 0 and batch > 0:												# If we want to print things out...
			cur_loss = total_loss[0] / DEUBG_LOG_INTERVAL												# Calculate the current loss
			elapsed = time.time() - start_time 															# Calculate how much time has passed for this epoch
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr ADAM | ms/batch {:5.2f} | '
					'loss {:5.2f} | ppl {:8.2f}'.format(
				epoch, batch, train_data.size()[1] // MAX_SEQ_LEN,
				elapsed * 1000 / DEUBG_LOG_INTERVAL, cur_loss, math.exp(cur_loss)))						# Print some log statement
			total_loss = 0																				# Reset the loss
			start_time = time.time()																	# Reset the time

# Loop over epochs.
lr = INITIAL_LEARNING_RATE  																			# Define our learning rate
best_val_loss = None																					# Define a best validation loss so we can do early stopping if we want to

optimizer = optim.RMSprop(model.parameters(), lr = INITIAL_LEARNING_RATE)								# Initialize the optimizer here, pass the parameters from the model so it can update them when its time, also initialize with the passed learning rate
optimizer.zero_grad()																					# Zero the gradient so it can be used from the start

# At any point you can hit Ctrl + C to break out of training early.
try:
	print("Starting training")
	for epoch in range(1, EPOCH_MAX+1):																	# For every epoch we are going to do
		epoch_start_time = time.time()																	# Set a start time
		train()																							# Train the model
		val_loss = evaluate(val_data)																	# test the model on validation data to check performance
		print('-' * 89)																					
		print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '							
				'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
										   val_loss, math.exp(val_loss)))								# Print some log statement
		print('-' * 89)
		
		if not best_val_loss or val_loss < best_val_loss:												# Save the model if the validation loss is the best we've seen so far.
			with open(MODEL_SAVE_PATH, 'wb') as f:
				torch.save(model, f)
			best_val_loss = val_loss

except KeyboardInterrupt:																				# If Ctrl + C was pressed
	print('-' * 89)
	print('Exiting from training early')																# Note that it was pressed and continue on

# Load the best saved model.
with open(MODEL_SAVE_PATH, 'rb') as f:																	# Load the saved model because training is done and we want the one that performed best on the validation data
	model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)																			# Evaluate the model on the test data
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
	test_loss, math.exp(test_loss)))																	# Print the model's performance on the testing data
print('=' * 89)
