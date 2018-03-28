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


    def forward(self, input, hidden, return_h=False):
        embedding = embed_regularize(self.encoder, input, dropout=DROPOUT_EMB if self.training else 0)

        dropped_embedding = self.lockdrop(embedding, DROPOUT_INP)

        hid_all = []

        rnns_before_drop = []
        rnns_after_drop = []

        for layer, rnn_unit in enumerate(self.rnns):
            if layer == 0:
            	rnn_out, rnn_hid = rnn_unit(dropped_embedding, hidden[layer])
            else:
            	rnn_out, rnn_hid = rnn_unit(rnn_out, hidden[layer])
            hid_all.append(rnn_hid)
            
            rnns_before_drop.append(rnn_out)
            if layer <= NUMBER_LAYERS -2:
                rnn_out = self.lockdrop(rnn_out, DROPOUT_HID)
                rnns_after_drop.append(rnn_out)

        final_out = self.lockdrop(rnn_out, DROPOUT)
        rnns_after_drop.append(final_out)
    
        d0 = final_out.size(0)
        d1 = final_out.size(1)
        d2 = final_out.size(2)
    
        final_out = final_out.view(d0*d1, d2)
        decoder_output = self.decoder(final_out)
        result = decoder_output.view(d0,d1, decoder_output.size(1))
        return result, hid_all , rnns_before_drop, rnns_after_drop

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE).zero_()),
                Variable(weight.new(1, bsz, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE).zero_()))
                for l in range(NUMBER_LAYERS)]
