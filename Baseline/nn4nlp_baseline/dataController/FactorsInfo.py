

class FactorsInfo:
	def __init__(self):
		self.__facs = {}
		self.__facsOrder = []

	def setFactorDetails(self, factorName, factorVocabSize, factorEmbeddingSize):
		self.__facsOrder.append(factorName)
		self.__facs[factorName] = (factorVocabSize, factorEmbeddingSize)

	def getFactor(self, factorName):
		return self.__facs[factorName]
	
	def getFactorVocabSize(self, factorName):
		return self.__facs[factorName][0]

	def getFactorEmbeddingSize(self, factorName):
		return self.__facs[factorName][1]

	def getFactorOrder(self):
		return self.__facsOrder

	def getConcatEmbeddingSize(self):
		res = 0
		for _, emb in self.__facs.values():
			res += emb
		return res

	def factorVocabSpan(self, factorName):
		offset = 0
		vocabSize = self.getFactorVocabSize(factorName)
		for factor in self.__facsOrder:
			if factor == factorName:
				return offset, offset+vocabSize
			else:
				offset += self.getFactorVocabSize(factor)
		return -1, -1

