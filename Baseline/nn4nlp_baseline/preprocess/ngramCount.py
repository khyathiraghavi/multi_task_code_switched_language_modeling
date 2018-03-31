def getNGrams(path, gramSize, factors):
	f = open(path, "r")
	lines = f.readlines()
	f.close()

	grams = {}

	for line in lines:
		words = line.split()
		for i in range(0, len(words)-gramSize):
			gram = words[i:i+gramSize]
			res = []
			for f in gram:
				if("LANG_FACTOR:HIN" in f):
					facs = f.split("/")
					for fac in facs:
						s = fac.split(":")
						if(s[0] in factors):
							res.append(s[1])
			key = " ".join(res)
			try:
				grams[key] += 1
			except KeyError as e:
				grams[key] = 1
	print "Found " + str(len(grams.keys()))+ " " + str(gramSize) + "-Grams in " + str("/".join(path.split("/")[-2:]))

def getCombos(path,factors):
	f = open(path, "r")
	lines = f.readlines()
	f.close()

	grams = {}

	for line in lines:
		words = line.split()
		for i in range(0, len(words)):
			gram = words[i]
			res = []
			facs = gram.split("/")
			for fac in facs:
				s = fac.split(":")
				if(s[0] in factors):
					res.append(s[1])


			try:
				grams[res[0]].append(res[1])
			except KeyError as e:
				grams[res[0]] = [res[1]]
	summation = 0
	for key, value in grams.items():
		summation += len(set(value))
	print float(summation)/(len(grams.items()))

def getNGramsCounts(path, gramSize, factor):
	f = open(path, "r")
	lines = f.readlines()
	f.close()

	grams = {}

	for line in lines:
		words = line.split()
		for i in range(0, len(words)-gramSize):
			gram = words[i:i+gramSize]
			res = []
			for f in gram:
				facs = f.split("/")
				for fac in facs:
					s = fac.split(":")
					if(s[0] == factor):
						res.append(s[1])
			key = " ".join(res)
			try:
				grams[key] += 1
			except KeyError as e:
				grams[key] = 1
	for key in grams.keys():
		print "Found " + str(grams[key])+ " instances of " + str(key) + " in " + str("/".join(path.split("/")[-2:]))


path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.train"
getNGrams(path, 1, ["WORD_FACTOR", "LANG_FACTOR"])

path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.test"
getNGrams(path, 1, ["WORD_FACTOR", "LANG_FACTOR"])

path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.val"
getNGrams(path, 1, ["WORD_FACTOR", "LANG_FACTOR"])
'''
path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.train"
getNGramsCounts(path, 1, "LANG_FACTOR")

path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.test"
getNGramsCounts(path, 1, "LANG_FACTOR")

path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.val"
getNGramsCounts(path, 1, "LANG_FACTOR")'''