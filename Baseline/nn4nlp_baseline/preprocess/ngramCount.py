def getNGrams(path, gramSize):
	

	f = open(path, "r")
	lines = f.readlines()
	f.close()

	grams = {}

	for line in lines:
		words = line.split()
		for i in range(0, len(words)-gramSize):
			gram = words[i:i+gramSize]
			key = " ".join(gram)
			try:
				grams[key] += 1
			except KeyError as e:
				grams[key] = 1
	print "Found " + str(len(grams.keys()))+ " " + str(gramSize) + "-Grams in " + str("/".join(path.split("/")[-2:]))


path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/nn4nlp_baseline/output/output.txt.train"
getNGrams(path, 1)
getNGrams(path, 2)
getNGrams(path, 3)
getNGrams(path, 4)
getNGrams(path, 5)
getNGrams(path, 6)

path = "/home/tmanzini/Thomas/workspace/nn4nlp_code-mixing/Baseline/pytorch_example/data/wikitext-2/train.txt"
getNGrams(path, 1)
getNGrams(path, 2)
getNGrams(path, 3)
getNGrams(path, 4)
getNGrams(path, 5)
getNGrams(path, 6)