import torch
import torch.nn as nn
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, nlang, langEmbSize, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, useLangEncoder=True, addLanguageBias = False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.word_encoder = nn.Embedding(ntoken, ninp)
        self.lang_encoder = nn.Embedding(nlang,  langEmbSize)

        if(useLangEncoder):
            self.rnnInputSize = ninp + langEmbSize
        else:
            self.rnnInputSize = ninp

        self.rnns = [torch.nn.LSTM(self.rnnInputSize if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=dropouth) for l in range(nlayers)]
        print(self.rnns)
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.langDecoder = nn.Linear(ninp, nlang)

        self.langDecoderBias = None
        if addLanguageBias:
            self.langDecoderBias = nn.Linear(nlang, ntoken)

        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.decoder.weight = self.word_encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.langEmbSize = langEmbSize
        self.useLangEncoder = useLangEncoder
        self.nhid = nhid
        self.nlang = nlang
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute

    def init_weights(self):
        initrange = 0.1
        self.word_encoder.weight.data.uniform_(-initrange, initrange)
        self.lang_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if not self.langDecoderBias is None:
            self.langDecoderBias.bias.data.fill_(0)
            self.langDecoderBias.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputWords, inputLang, hidden, return_h=False):
        wordEmb = embedded_dropout(self.word_encoder, inputWords, dropout=self.dropoute if self.training else 0)
        wordEmb = self.lockdrop(wordEmb, self.dropouti)

        langEmb = embedded_dropout(self.lang_encoder, inputLang, dropout=self.dropoute if self.training else 0)
        langEmb = self.lockdrop(langEmb, self.dropouti)

        if(self.useLangEncoder == True):
            raw_output = torch.cat((wordEmb, langEmb), 2)
        else:
            raw_output = wordEmb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)


        predBasis = output.view(output.size(0)*output.size(1), output.size(2))

        langPred = self.langDecoder(predBasis)

        if not self.langDecoderBias is None:
            biasTerm = self.langDecoderBias(langPred)
            self.decoder.bias.data = biasTerm.data

        decoded = self.decoder(predBasis)
        
        result     = decoded.view(output.size(0), output.size(1), decoded.size(1))
        langResult = langPred.view(output.size(0), output.size(1), langPred.size(1))

        if return_h:
            return result, langResult, hidden, raw_outputs, outputs
        return result, langResult, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()),
                Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.ninp).zero_()))
                for l in range(self.nlayers)]
