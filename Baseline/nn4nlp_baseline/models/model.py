import torch
import torch.nn as nn
from torch.autograd import Variable

class FactoredRNN(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, factorsInfo, outputVocabSize, nhid, nlayers, dropout=0.5):
		super(FactoredRNN, self).__init__()
		self.factorsInfo = factorsInfo

		self.drop = nn.Dropout(dropout)		
		self.encoders = []
		for factor in factorsInfo.getFactorOrder():
			fVocabSize, fEmbeddingSize = factorsInfo.getFactor(factor)
			fEnc = nn.Embedding(fVocabSize, fEmbeddingSize)
			self.encoders.append(fEnc)

		rnnInputSize = factorsInfo.getConcatEmbeddingSize()

		if rnn_type in ['LSTM', 'GRU']:														
			self.rnn = getattr(nn, rnn_type)(rnnInputSize, nhid, nlayers, dropout=dropout, batch_first=True)			
		else:																				
			try:																			
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]			
			except KeyError:																
				raise ValueError( """An invalid option for `--model` was supplied,				
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")	
			self.rnn = nn.RNN(rnnInputSize, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
		self.decoder = nn.Linear(nhid, outputVocabSize)												

		self.init_weights()																	

		self.rnn_type = rnn_type															
		self.nhid = nhid																	
		self.nlayers = nlayers																


	#This is the function that is called to initalize the weights to some starting values
	#This function takes no arguments and 
	def init_weights(self):
		initrange = 0.1																			#Define a range that you want to initialize the weight matrix
		for encoder in self.encoders:
			encoder.weight.data.uniform_(-initrange, initrange)									#Initialize the weights for the encoder to a matrix filled with numbers drawn from a uniformly random distribution along the predefined range, note that embedding layers do not have bias terms so that is why we dont handle initialization of bias terms for the encoder
																								#Note that the "_" postscript of the uniform function means that this operation is done in place and the value is not returned
		self.decoder.bias.data.fill_(0)															#Set the bias on the decoder to all zeros
		self.decoder.weight.data.uniform_(-initrange, initrange)								#initialize the weights for the decoder to a matrix filled with numbers drawn from a uniformly random distribution along the predefined range

	#This is the function that defines the forward pass of our model and actually generates an output
	#It takes 2 values as an input, the first is the actual input to the model, the second is the hidden layer of the RNN.
	#Input - This is the actual input to our network. This is a matrix which is shape (max seq len)x(batch size)
	#Hidden - This represents the hidden value in our RNN cell after it has completed its iterations, it is taken as an argument in all RNN cells so this is just keeping up the design pattern, but this is because you may want to initialize it in a special way or something
	def forward(self, input, hidden):
		inputs = []
		for i in range(0, input.size()[-1]):
			inputs.append(input[:, :, i].contiguous().view(input.size()[0], input.size()[1]))
		embs = []
		for enc, inp in zip(self.encoders, inputs):
			embs.append(self.drop(enc(inp)))													#Calculate the embedding value by extracting the embedding for the words in input and then performing dropout

		concatEmb = torch.cat(embs, 2)

		output, hidden = self.rnn(concatEmb, hidden)											#Calculate the output of the RNN by passing in the embedding value from the previous layer and the hidden state
		output = self.drop(output) 																#Perform dropout on the output, this is done seperately so we can keep track of the hidden state
		
		#NOTE THIS PART IS SUPER CONFUSING, ILL DO MY BEST TO EXPLAIN
		#This code here: output.view(output.size(0)*output.size(1), output.size(2)) is doing a reshape, thats what the view() function does in pytorch
		#But now the question is why are we doing a reshape??? Heres why... We are passing the data to the linear data independent of sequence... We have 2 things going on here, 
		#we have batches and we have a sequences of classifications in those batches. Remember this is a generic RNN so we are still generating outputs for every time we step 
		#through the RNN cell. So because we are not training a different linear layer for every step in the output and we are not training a different linear layer for every 
		#batch, we can squish all of these examples together and have the linear layer do its operation all at once. 
		#So this view (reshape) is effectively removing the sequential nature of the data and then passing it to the RNN as a whole bunch of non-sequential training examples. 
		#So the linear layer is computing an output for every output of the RNN for every batch as if they were indepenedent of their sequence
		#This step is effectively an optimization step

		decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

		return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden 			#Once we have that output, we do the reverse of the reshape to get everything back in to the sequential form, and then we return it

	#This function initializes the hidden states of the RNN for when we want to first start the predictions
	#It takes one argument and that is bsz, which is an annoying way of writing batchsize
	def init_hidden(self, bsz):
		weight = next(self.parameters()).data										#This is an esoteric way of saying get the parameters for the RNN cell... 
																					#Next() gets the next value from an iterator, and paramters() is an entry point into the parameters of the model. 
																					#Our zeroth paramters are the embedding layer, the first is the RNN cell... this is dumb but this is the way that pytorch deicded to write this 
																					#the X.data means get the actual data as a tensor 
		if self.rnn_type == 'LSTM':													#If we are using LSTMs as our RNN cells
			return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),		
					Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))		#There are two hidden states, one for the "cell state" and one for the "hidden state" so initalize them both the zeros and wrap them as variables (variables can be auto-differntiated while tensors cannot)
																					#New() constructs a tensor of the same type as the source tensor
		else:																		#If we arent using LSTMs
			return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())		#Then we only have one hidden thing to worry about, so initialize it to zeros 

	#Note we dont have to define a backwards function as it is predefined by the framework based on the forward function. pytorch keeps track of what operations are done 
	#to which matricies and does the differentiation for us so we dont have to.
