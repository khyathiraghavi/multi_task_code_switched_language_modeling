import torch.nn as nn
from torch.autograd import Variable


#This is the RNNModel that we are going to be exploring today, this class is written as a generic RNN, so there will be an output that is returned for every input despite 
# it being used for a language modeling task that only requires one output at the final cell. This will come into play when doing the forward propagation, so dont get 
# confused if you see that the model doing things that you wouldnt expect for a language modeling task
class CachedRNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	#This is the constructor, it takes several arguments... 
	#rnn_type - a poorly defined variable name in this example, it either defines the type of cell that is going to be used, or the nonlinearity in a simple RNN cell
	#           The options for this variable are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']
	#ntoken - the size of your vocabulary
	#ninp - NOT the size of your input sequence as the name would imply, this is the dimesion of your embeddings that you are going to learn
	#nhid - the size of the hidden layer inside your RNN cell
	#nlayers - the number of layers in size your RNN cell
	#dropout - the percent of weights that are going to be dropped out during training
	#tie_weights - a boolean variable that defines if your encoder weights from the embedding layer and the decoder from the linear layer are going to be the same... 
	#              This may seem odd but it basically requires that your embedding layer IS your decoding layer. So it requires that your decoding layer have the same
	#              shape as your embedding layer. Since in both the embedding and decoding layer you have one axis that is the vocabulary size that is good, so you just
	#              need to set the hidden size to the embedding size. Apparently this has good performance (https://arxiv.org/abs/1608.05859, https://arxiv.org/abs/1611.01462)
	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
		super(CachedRNNModel, self).__init__()													#Call the RNN constructor
		self.drop = nn.Dropout(dropout)															#Define a dropout layer with dropout rate "dropout"
		self.encoder = nn.Embedding(ntoken, ninp)												#Define an encoder which is the embedding layer, it has embedding size ninp and ntoken number of embeddings (everything else should be unks)
		if rnn_type in ['LSTM', 'GRU']:															#If the rnn_type is specified as LSTMs or GRUs
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)				#Then get the LSTM/GRU class from the nn module, and pass the args into the constructors - the 2 constructors have the same parameters so you can do this... though its bad programming style
		else:																					#If the passed RNN type isnt a LSTM or GRU
			try:																				#Try the following					
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]				#Check to see if its RNN_RELU/RNN_TANH, and then define the nonlinearity in terms of what is needed for the RNN constructor
			except KeyError:																	#If it fails then the user has passed an invalid RNN Cell type
				raise ValueError( """An invalid option for `--model` was supplied,				
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")		#So we throw an error
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)	#If we havent thrown an error, then we create an RNN cell and call it our RNN with the appropriate args
		self.decoder = nn.Linear(nhid, ntoken)													#Then we define a linear layer which is our decoder. Presumably you could do a softmax over this linear layer here, im not sure why they arent doing that already 

		if tie_weights:																			#If you are going to tie the weights...
			if nhid != ninp:																	#Ensure that the dimensions of the encoder and the decoder are the same
				raise ValueError('When using the tied flag, nhid must be equal to emsize')		#If they arent then throw an error
			self.decoder.weight = self.encoder.weight 											#Then actually tie the weights by setting the weights of the decoder to the weights of the encoder... this is a pass by reference so you arent setting the value, instead you are saying that the weights should be tied

		self.init_weights()																		#Initialize the weights, see documentation for the function to see whats up here

		self.rnn_type = rnn_type																#Set the rnn_type member variable
		self.nhid = nhid																		#Set the nhid member variable
		self.nlayers = nlayers																	#Set the nlayers member variable


	#This is the function that is called to initalize the weights to some starting values
	#This function takes no arguments and 
	def init_weights(self):
		initrange = 0.1																			#Define a range that you want to initialize the weight matrix
		self.encoder.weight.data.uniform_(-initrange, initrange)								#Initialize the weights for the encoder to a matrix filled with numbers drawn from a uniformly random distribution along the predefined range, note that embedding layers do not have bias terms so that is why we dont handle initialization of bias terms for the encoder
																								#Note that the "_" postscript of the uniform function means that this operation is done in place and the value is not returned
		self.decoder.bias.data.fill_(0)															#Set the bias on the decoder to all zeros
		self.decoder.weight.data.uniform_(-initrange, initrange)								#initialize the weights for the decoder to a matrix filled with numbers drawn from a uniformly random distribution along the predefined range

	#This is the function that defines the forward pass of our model and actually generates an output
	#It takes 2 values as an input, the first is the actual input to the model, the second is the hidden layer of the RNN.
	#Input - This is the actual input to our network. This is a matrix which is shape (max seq len)x(batch size)
	#Hidden - This represents the hidden value in our RNN cell after it has completed its iterations, it is taken as an argument in all RNN cells so this is just keeping up the design pattern, but this is because you may want to initialize it in a special way or something
	def forward(self, input, hidden):
		emb = self.drop(self.encoder(input))													#Calculate the embedding value by extracting the embedding for the words in input and then performing dropout
		output, hidden = self.rnn(emb, hidden)													#Calculate the output of the RNN by passing in the embedding value from the previous layer and the hidden state
		output = self.drop(output) 																#Perform dropout on the output, this is done seperately so we can keep track of the hidden state
		
		print output.size()


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

		decodedSeqs = decoded.view(output.size(0), output.size(1), decoded.size(1))				#Once we have that output, we do the reverse of the reshape to get everything back in to the sequential form,

		return decodedSeqs, hidden 																#And then we return it

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
