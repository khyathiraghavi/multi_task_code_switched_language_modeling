import torch
import torch.nn as nn
from torch.autograd import Variable

class locked_dropout(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, weight, dropout=0.5):
		if not self.training or not dropout:
			return weight
		scale_lock_drop_mask = weight.data.new(weight.size(0), 1, weight.size(2)).bernoulli_(1 - dropout)/(1 - dropout)
		lock_drop_var = Variable(scale_lock_drop_mask) # requires_grad defaults to false
		lock_drop_mask = lock_drop_var.expand_as(weight)
		return lock_drop_mask * weight
