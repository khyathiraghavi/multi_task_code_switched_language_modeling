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
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        #self.init_weights()
        
        self.encoder.weight.data.uniform_(-WEIGHT_INIT, WEIGHT_INIT)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-WEIGHT_INIT, WEIGHT_INIT)

        
        '''
        self.rnn_type = MODEL_TYPE
        self.ninp = WORD_EMBEDDING_SIZE
        self.nhid = NUMBER_HIDDEN
        self.nlayers = NUMBER_LAYERS
        self.dropout = DROPOUT
        self.dropouti = DROPOUT_INP
        self.dropouth = DROPOUT_HID
        self.dropoute = DROPOUT_EMB
        '''

        '''
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        '''

    '''
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    '''

    def forward(self, input, hidden, return_h=False):
        emb = embed_regularize(self.encoder, input, dropout=DROPOUT_EMB if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, DROPOUT_INP)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != NUMBER_LAYERS - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, DROPOUT_HID)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, DROPOUT)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE).zero_()),
                Variable(weight.new(1, bsz, NUMBER_HIDDEN if l != NUMBER_LAYERS - 1 else WORD_EMBEDDING_SIZE).zero_()))
                for l in range(NUMBER_LAYERS)]
