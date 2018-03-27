from constants import *
import torch
import torch.nn as nn
from torch.autograd import Variable

from optimization.embed_regularize import embed_regularize
from optimization.locked_dropout import locked_dropout
from optimization.weight_drop import weight_drop

class RNNModel(nn.Module):

	def __init__(self, num_token):
		super(RNNModel, self).__init__()
		self.lockdrop = locked_dropout()
		self.encoder = nn.Embedding(num_token, WORD_EMBEDDING_SIZE)
		self.rnns = [torch.nn.LSTM(WORD_EMBEDDING_SIZE if l == 0 else NUMBER_HIDDEN, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE, 1, dropout=DROPOUT_HID) for l in range(NUMBER_LAYERS)]
		print(self.rnns)
		if WEIGHT_DROP:
			self.rnns = [weight_drop(rnn, ['weight_hh_l0'], dropout=WEIGHT_DROP) for rnn in self.rnns]
		self.rnns = torch.nn.ModuleList(self.rnns)
		self.decoder = nn.Linear(NUMBER_HIDDEN, num_token)

		if TIED:
			self.decoder.weight = self.encoder.weight

		
		self.encoder.weight.data.uniform_(-WEIGHT_INIT, WEIGHT_INIT)
		self.decoder.bias.data.fill_(0)
		self.decoder.weight.data.uniform_(-WEIGHT_INIT, WEIGHT_INIT)


	def forward(self, input, hidden, return_h=False):
		emb = embed_regularize(self.encoder, input, dropout=DROPOUT_EMB if self.training else 0)

		emb = self.lockdrop(emb, DROPOUT_INP)

		raw_output = emb
		new_hidden = []

		raw_outputs = []
		outputs = []
		for layer, rnn_unit in enumerate(self.rnns):
			raw_output, new_h = rnn_unit(raw_output, hidden[layer])
			new_hidden.append(new_h)
			raw_outputs.append(raw_output)
			if layer <= NUMBER_LAYERS -2:
				raw_output = self.lockdrop(raw_output, DROPOUT_HID)
				outputs.append(raw_output)
		hidden = new_hidden

		output = self.lockdrop(raw_output, DROPOUT)
		outputs.append(output)
	
		d0 = output.size(0)
		d1 = output.size(1)
		d2 = output.size(2)
	
		output = output.view(d0*d1, d2)
		decoder_output = self.decoder(output)
		result = decoder_output.view(d0,d1, decoder_output.size(1))
		#if return_h:
		return result, hidden, raw_outputs, outputs
		#return result, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		return [(Variable(weight.new(1, bsz, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE).zero_()),
				Variable(weight.new(1, bsz, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE).zero_()))
				for l in range(NUMBER_LAYERS)]
