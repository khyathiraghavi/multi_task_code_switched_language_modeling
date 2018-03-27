import os
import torch

from collections import Counter


class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
		self.counter = Counter()
		self.total = 0

	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		token_id = self.word2idx[word]
		self.counter[token_id] += 1
		self.total += 1
		return self.word2idx[word]

	def __len__(self):
		return len(self.idx2word)


class Corpus(object):
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))

	def tokenize(self, path):
		"""Tokenizes a text file."""
		#assert os.path.exists(path)
		# Add words to the dictionary
		with open(path, 'r') as f:
			lines = f.readlines()
			num_tokens = 0
			for line in lines:
				words = line.strip().split() + ['<eos>']
				num_tokens += len(words)
				for word in words:
					self.dictionary.add_word(word)

			#with open(path, 'r') as f:
			ids = torch.LongTensor(num_tokens)
			token_idx = 0
			for line in lines:
				words = line.strip().split() + ['<eos>']
				for word in words:
					ids[token_idx] = self.dictionary.word2idx[word]
					token_idx += 1

		return ids

