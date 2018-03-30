import torch
from torch.autograd import Variable


def embed_regularize(embed, words, dropout=0.1):
  masked_embed_weight = embed.weight
  if dropout:
    # a.bernoulli_(x) is equivalent to torch.bernoulli(x, out=a) where x is the bernoulli prob. and a is the output tensor
    # expand_as is eqv to self.expand(tensor.size())
    embed_dropout_mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight)
    scaled_dropout_mask = embed_dropout_mask/(1 - dropout)
    dropout_var = Variable(scaled_dropout_mask)
    masked_embed_weight = dropout_var * embed.weight

  padding_idx = embed.padding_idx
  if padding_idx is None:
    padding_idx = -1
  X = embed._backend.Embedding(padding_idx, embed.max_norm, embed.norm_type,
    embed.scale_grad_by_freq, embed.sparse
    )(words, masked_embed_weight)
  return X
