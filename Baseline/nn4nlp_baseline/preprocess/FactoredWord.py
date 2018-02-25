class FactoredWord:
	def __init__(self, factorList):
		self.__factors = {}
		for factor in factorList:
			self.__factors[factor] = None

	def addFactor(self, field, value):
		self.__factors[field] = value

	def getFactor(self, field):
		return self.__factors[field]
	def __str__(self):
		facs = [str(key) + ":" + str(value) for key, value in self.__factors.items()]
		return "/".join(facs)
	def __eq__(self, b):
		if(self.getWord() != b.getWord()):
			return False
		for field in self.__factors.keys():
			if(self.getFactor(field) != b.getFactor(field)):
				return False
		return True