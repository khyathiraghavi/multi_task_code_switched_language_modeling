import sys

class Counts:
	def __init__(self):
		self.__wordCounts = {}
		self.__ranks = None
		self.__inVocab = {}
	def addWord(self, newWord):
		self.__wordCounts[newWord] = 0
	def incrementWord(self, word, addWord=True, amount=1):
		try:
			self.__wordCounts[word] += amount
		except KeyError as e:
			if (not addWord):
				raise e
			else:
				self.__wordCounts[word] = amount
	def getWordCount(self, word):
		return self.__wordCounts[word]
	def getTopK(self, k):
		return [i[0] for i in sorted(self.__wordCounts.items(), key=lambda x:-x[1])[:k]]
	def fixRanks(self):
		self.__ranks = {}
		ranked = sorted(self.__wordCounts.items(), key=lambda x:-x[1])
		for i, word in enumerate(ranked):
			self.__ranks[word[0]] = i

	def getRank(self, word):
		if(self.__ranks == None):
			raise Exception("Call fixRanks() before accessing ranks")
		try:
			return self.__ranks[word]
		except KeyError as e:
			return sys.maxint
	def __len__(self):
		return len(self.__wordCounts.keys())