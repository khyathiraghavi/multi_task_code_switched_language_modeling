from constants import *
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from optimization.embed_regularize import embed_regularize
from optimization.locked_dropout import locked_dropout
from optimization.weight_drop import weight_drop

class RNNModel(nn.Module):

    def __init__(self, num_token):
        super(RNNModel, self).__init__()
        self.locked_drop = locked_dropout()
        self.encoder = nn.Embedding(num_token, WORD_EMBEDDING_SIZE)
        self.rnns = []
        self.rnns.append(getattr(nn, MODEL_TYPE)(WORD_EMBEDDING_SIZE, NUMBER_HIDDEN,1, dropout=DROPOUT_HID))
        self.rnns.append(getattr(nn, MODEL_TYPE)(NUMBER_HIDDEN, NUMBER_HIDDEN, 1, dropout=DROPOUT_HID))
        self.rnns.append(getattr(nn, MODEL_TYPE)(NUMBER_HIDDEN, WORD_EMBEDDING_SIZE, 1, dropout=DROPOUT_HID))
        if WEIGHT_DROP:
            self.rnns = [weight_drop(rnn, ['weight_hh_l0'], dropout=WEIGHT_DROP) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(NUMBER_HIDDEN, num_token)

        if TIED:
            self.decoder.weight = self.encoder.weight

        
        self.encoder.weight.data.uniform_(-WEIGHT_INIT, WEIGHT_INIT)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-WEIGHT_INIT, WEIGHT_INIT)

        self.hid_all = []
        self.rnns_before_drop = []
        self.rnns_after_drop = []
        self.ar_fragment = 0
        self.tar_fragment = 0


    def forward(self, input, hidden):
        embedding = embed_regularize(self.encoder, input, dropout=DROPOUT_EMB if self.training else 0)

        dropped_embedding = self.locked_drop(embedding, DROPOUT_INP)

        self.hid_all = []

        self.rnns_before_drop = []
        self.rnns_after_drop = []

        for layer, rnn_unit in enumerate(self.rnns):
            if layer == 0:
                rnn_out, rnn_hid = rnn_unit(dropped_embedding, hidden[layer])
            else:
                rnn_out, rnn_hid = rnn_unit(rnn_out, hidden[layer])
            self.hid_all.append(rnn_hid)
            
            self.rnns_before_drop.append(rnn_out)
            if layer <= NUMBER_LAYERS -2:
                rnn_out = self.locked_drop(rnn_out, DROPOUT_HID)
                self.rnns_after_drop.append(rnn_out)

        final_out = self.locked_drop(rnn_out, DROPOUT)
        self.rnns_after_drop.append(final_out)
    
        self.rnns_before_drop = self.rnns_before_drop[-1:]
        self.rnns_after_drop = self.rnns_after_drop[-1:]
        d0 = final_out.size(0)
        d1 = final_out.size(1)
        d2 = final_out.size(2)
    
        self.ar_fragment =  sum( el.pow(2).mean() for el in self.rnns_after_drop)
        self.tar_fragment =  sum( (el[1:]-el[:-1]).pow(2).mean() for el in self.rnns_before_drop)
        final_out = final_out.view(d0*d1, d2)
        decoder_output = self.decoder(final_out)
        result = decoder_output.view(d0,d1, decoder_output.size(1))
        return result, self.hid_all
    
    def init_hidden(self,bsz):
        weight = next(self.parameters()).data
        hidden_initialization = []
        for layer in range(NUMBER_LAYERS):
            if layer == NUMBER_LAYERS-1:
                hidden_initialization.append( ( autograd.Variable(weight.new(1, bsz, WORD_EMBEDDING_SIZE).zero_()),  \
                                               autograd.Variable(weight.new(1, bsz, WORD_EMBEDDING_SIZE).zero_())  ) )
            else:
                hidden_initialization.append( ( autograd.Variable(weight.new(1, bsz, NUMBER_HIDDEN).zero_()), \
                                               autograd.Variable(weight.new(1, bsz, NUMBER_HIDDEN).zero_())  ) )
        return hidden_initialization






