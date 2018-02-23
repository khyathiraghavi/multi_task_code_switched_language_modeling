UNKOWN_FACTOR = "UNK_FACTOR"

class FactoredWord:
	def __init__(self, word):
		self.__word = word
		self.__factors = {}

	def addFactor(self, field, value):
		self.__factors[field] = value

	def setWord(self, newWord):
		self.__word = newWord

	def getWord(self):
		return self.__word

	def getFactor(self, field):
		try:
			return self.__factors[field]
		except:
			return str(field)+"_"+UNKOWN_FACTOR
	def __str__(self):
		return "/".join([self.__word] + self.__factors.values())
	def __eq__(self, b):
		if(self.getWord() != b.getWord()):
			return False
		for field in self.__factors.keys():
			if(self.getFactor(field) != b.getFactor(field)):
				return False
		return True