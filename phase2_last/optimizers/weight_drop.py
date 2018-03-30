import torch
from torch.nn import Parameter
from functools import wraps

class weight_drop(torch.nn.Module):
	def __init__(self, module, weights, dropout=0):
		super(weight_drop, self).__init__()
		self.module = module
		self.weights = weights
		self.dropout = dropout
		self.configureDropout()

	def configureDropout(self):
		for weight_params in self.weights:
			w = getattr(self.module, weight_params)
			del self.module._parameters[weight_params]
			print('Configuring drop out of {} to parameter {}'.format(self.dropout, weight_params))
			self.module.register_parameter(weight_params + '_raw', Parameter(w.data))

	def performDropout(self):
		for weight_params in self.weights:
			w = getattr(self.module, weight_params + '_raw')
			w = torch.nn.functional.dropout(w, p=self.dropout, training=self.training)
			setattr(self.module, weight_params, w)

	def forward(self, *args):
		self.performDropout()
		return self.module.forward(*args)
