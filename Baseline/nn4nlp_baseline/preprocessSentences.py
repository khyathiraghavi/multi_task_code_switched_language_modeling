import nltk

from Counts import Counts
from FactoredWord import FactoredWord
from random import shuffle

VOCAB_SIZE = 5000
MAX_SENTENCE_LEN = 100
MAKE_UPPER = True
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

LANGAUGE_FACTOR = "LANG_FACTOR"

TRAIN_PERCENT = 0.8
TEST_PERCENT = 0.1
VALIDATION_PERCENT = 0.1

INPUT_FILENAME = "demoData/mixed_input_ratio_20.txt"

OUTPUT_FILENAME = "output.txt"

def cleanLine(toClean):
	return toClean.upper().replace("\n", "").replace("\r", "").replace("\t", " ").lstrip(" ").rstrip(" ")

def tokenize(toTokenize):
	toks = toTokenize.split(" ")[1:min(MAX_SENTENCE_LEN, len(toTokenize)-1)]
	res = []
	for token in toks:
		word, lang = token.split("\\")
		fWord = FactoredWord(word)
		fWord.addFactor(LANGAUGE_FACTOR, lang)
		res.append(fWord)
	res.append(FactoredWord(EOS_TOKEN))
	return res

def unkify(toUnk, mc):
	res = []
	for token in toUnk:
		if (mc.getRank(token.getWord()) > VOCAB_SIZE):
			token.setWord(UNK_TOKEN)
		res.append(token)
	return res

f = open(INPUT_FILENAME, "r")
lines = f.readlines()
f.close()
print "Read file"

#CLEAN AND TOKENIZE THE LINES
cleanLines = []
for line in lines:
	cleanedLine = cleanLine(line)
	cleanLines.append(tokenize(cleanedLine))
print "cleaned lines"

#GET THE COUNTS FROM THE SENTENCES
mc = Counts()
for line in cleanLines:
	for token in line:
		mc.incrementWord(token.getWord())
mc.fixRanks()
print "Counted the words"
print "Number of words in the dataset: ", len(mc)

#REMOVE ALL OF THE UNKNOWN WORDS
unkedLines = []
for i, line in enumerate(cleanLines):
	unkedLines.append(unkify(line, mc))
print "unked the data"

#SHUFFLE THE LINES FOR OUTPUTS
shuffledLines = unkedLines
shuffle(shuffledLines)
print "Shuffled the data"

#Split the lines into training, validation, and test splits
N = len(shuffledLines)
trainSentences      = shuffledLines[:int(N*TRAIN_PERCENT)]
testingSentences    = shuffledLines[int(N*TRAIN_PERCENT+1):int(N*TRAIN_PERCENT+N*VALIDATION_PERCENT)]
validationSentences = shuffledLines[int(N*TRAIN_PERCENT+N*VALIDATION_PERCENT+1):]

f = open(OUTPUT_FILENAME+".train", "w")
for line in trainSentences:
	f.write(" ".join([str(word) for word in line]) + "\n")
f.close()

f = open(OUTPUT_FILENAME+".val", "w")
for line in validationSentences:
	f.write(" ".join([str(word) for word in line]) + "\n")
f.close()

f = open(OUTPUT_FILENAME+".test", "w")
for line in testingSentences:
	f.write(" ".join([str(word) for word in line]) + "\n")
f.close()
print "Done"


