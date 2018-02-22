import nltk

from Counts import Counts

VOCAB_SIZE = 500
MAX_SENTENCE_LEN = 100
MAKE_UPPER = True
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

TRAIN_PERCENT = 0.8
TEST_PERCENT = 0.1
VALIDATION_PERCENT = 0.1

OUTPUT_FILENAME = "output.txt"

def cleanLine(toClean):
	return toClean.upper().replace("\n", "").replace("\r", "").replace("\t", " ")

def tokenize(toTokenize):
	res = nltk.word_tokenize(toTokenize)
	res.append(EOS_TOKEN)
	return res
 

f = open("test.txt", "r")
lines = f.readlines()
f.close()

finalLines = []
for line in lines:
	cleanedLine = cleanLine(line)
	finalLines.append(tokenize(cleanedLine))

mc = Counts()
for line in finalLines:
	for token in line:
		mc.incrementWord(token)
mc.fixRanks()

print mc.getTopK(5)
print [mc.getRank(i) for i in mc.getTopK(5)]
