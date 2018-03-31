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

def oneHotify(hotIndex, size, datatype=np.float32, cuda=True):
	oneHot = np.zeros((1, size), datatype)
	oneHot[0][hotIndex] = 1
	result = Variable(torch.from_numpy(oneHot))
	if cuda:
		return result.cuda()
	else:
		return result


def evaluate(data):

	wordCache = None
	hiddenCache = None
	windowStartIndex = None
	
	model.eval()
	totalLoss = 0
	uncachedHiddenState = model.init_hidden(bsz)
	for i in range(0, data.size(0) - 1, BPTT):
		
		X, Y = batching.get_batch(data, i, evaluation=True)
		output, uncachedHiddenState = model(X, uncachedHiddenState)
		predictions = output.view(-1, vocabSize)
		outerMostHidden = model.rnns_before_drop[-1].squeeze()

		#Set our starting position for the window that will keep track of the cache
		#Update the words in the cache based on the one hot vectors for the targets
		#Update the hidden states in the cache based on what has been generated before
		oneHots = [oneHotify(label.data[0], vocabSize) for label in Y]
		hiddenValuesToCache = Variable(outerMostHidden.data)
		#If the cache hasnt been initialized yet...
		if wordCache == None:
			wordCache = torch.cat([oneHots])
			hiddenCache = hiddenValuesToCache
			windowStartIndex = 0
		#If the cache has been initialized...
		else:
			wordCache = torch.cat([wordCache, oneHots])
			hiddenCache = torch.cat([hiddenCache, hiddenValuesToCache], dim=0)
			windowStartIndex = len(wordCache)

		currentLoss = 0
		softmaxOutputs = torch.nn.functional.softmax(predictions)
		for wordIndex, modelProbs in enumerate(softmaxOutputs):

			#If we dont have the cache (as determined by the if statement) then we still need to have a distribution to draw from
			finalProbs = modelProbs

			#If we are outside the cache (we always should be) use the cache
			#This can also be seen as... if the cache has been constructed
			if windowStartIndex + wordIndex > CACHE_WINDOW_SIZE:
				#Construct the window of the cache that we are going to be operating over
				slicedWordCache   =   wordCache[windowStartIndex + wordIndex - CACHE_WINDOW_SIZE:windowStartIndex + wordIndex]
				slicedHiddenCache = hiddenCache[windowStartIndex + wordIndex - CACHE_WINDOW_SIZE:windowStartIndex + wordIndex]

				#Construct a vector of values that describe how well outerMostHidden correlates with the hidden values in the cache 
				hiddenCorrelation = torch.mv(slicedHiddenCache, outerMostHidden)

				#Pass the correlation values through a softmax so we can think of them as probabilities
				hiddenProbs = nn.softmax(THETA * hiddenCorrelation).view(-1, 1)

				#Calculate cache probabilities based on the probs from the softmax above times the one hot vectors we calculated earlier. 
				#As the values in slicedWordCache are one hot vectors this will not change the nature of this distribution
				cacheProbs = hiddenProbs.expand_as(slicedWordCache) * slicedWordCache.sum(0).squeeze()

				#Calculate the combined probabilities for the cache and the model based on a linear interpolation
				finalProbs = LAMBDA * cacheProbs + (1-LAMBDA) * modelProbs

			probOfTargetWord = finalProbs[Y[wordIndex]].data
			currentLoss += (-torch.log(probOfTargetWord)).data[0]
		totalLoss += currentLoss/TEST_BATCH_SIZE
		
		uncachedHiddenState = repackage_hidden(uncachedHiddenState)
		wordCache = wordCache[-CACHE_WINDOW_SIZE:]
		hiddenCache = hiddenProbs[-CACHE_WINDOW_SIZE:]

	final_loss = totalLoss[0] / len(data)
	print("Evaluation - Loss: " + str(final_loss) + " Perplexity: " + str(math.exp(final_loss)))

	return final_loss



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
	parser.add_argument('--cuda', action='store_false', help='use CUDA')
	parser.add_argument('--save', type=str,default='best.pt', help='model to use the pointer over')
	args = parser.parse_args()

	torch.manual_seed(RNG_SEED)
	if torch.cuda.is_available():
		if not args.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")
		else:
			torch.cuda.manual_seed(RNG_SEED)

	TEST_BATCH_SIZE = 1

	##### LOADING DATA #####

	corpus = data.Corpus(DATA_PATH)
	# retaining the same batch size to reproduce the results.
	batching = Batching()
	val_data = batching.batchify(corpus.valid)
	test_data = batching.batchify(corpus.test, TEST_BATCH_SIZE, args)

	print ("done batchifying")

	vocabSize = len(corpus.dictionary)
	model = model.RNNModel(vocabSize)

	# Load the best saved model.
	with open(args.save, 'rb') as f:
		if not args.cuda:
			model = torch.load(f, map_location=lambda storage, loc: storage)
		else:
			model = torch.load(f)
			model.cuda()
	print(model)

	# Run on val data.
	valLoss = evaluate(val_data)
	print('=' * 89)
	print('| End of pointer | val loss {:5.2f} | val ppl {:8.2f}'.format(
		valLoss, math.exp(valLoss)))
	print('=' * 89)

	# Run on test data.
	testLoss = evaluate(test_data)
	print('=' * 89)
	print('| End of pointer | test loss {:5.2f} | test ppl {:8.2f}'.format(
		testLoss, math.exp(testLoss)))
	print('=' * 89)
