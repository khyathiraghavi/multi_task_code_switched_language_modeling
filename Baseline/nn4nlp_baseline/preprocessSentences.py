import nltk

from Counts import Counts
from FactoredWord import FactoredWord
from random import shuffle

VOCAB_SIZE = 5000
LANG_ID_VOCAB_SIZE = 5
SOUNDEX_VOCAB_SIZE = 2000
MAX_SENTENCE_LEN = 100
MAKE_UPPER = True
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

LANGAUGE_FACTOR = "LANG_FACTOR"
SOUNDEX_FACTOR = "SOUNDEX_FACTOR"

TRAIN_PERCENT = 0.8
TEST_PERCENT = 0.1
VALIDATION_PERCENT = 0.1

INPUT_FILENAME = "demoData/sentences.txt"
SOUNDEX_INPUT_FILE = "demoData/soundex.txt"
LANG_INPUT_FILE = "demoData/lang_ids.txt"

OUTPUT_FILENAME = "output/output.txt"

def cleanLine(toClean):
	return toClean.upper().replace("\n", "").replace("\r", "").replace("\t", " ").lstrip(" ").rstrip(" ")

def tokenize(toTokenize, lanIDsToTokenize, soundexToTokenize):
	splits        =        toTokenize.split(" ")
	soundexSplits = soundexToTokenize.split(" ")
	langSplits    =  lanIDsToTokenize.split(" ")

	toks        =        splits[0:min(MAX_SENTENCE_LEN, len(splits))]
	langToks    =    langSplits[0:min(MAX_SENTENCE_LEN, len(splits))]
	soundexToks = soundexSplits[0:min(MAX_SENTENCE_LEN, len(splits))]
	
	assert(len(toks) == len(soundexToks) and len(soundexToks) == len(langToks)) #Make sure that soundex aligns with the tokens
	
	res = []
	for token, langToken, soundToken in zip(toks, langToks, soundexToks):
		fWord = FactoredWord(token)
		fWord.addFactor(LANGAUGE_FACTOR, langToken)
		fWord.addFactor(SOUNDEX_FACTOR, soundToken)
		res.append(fWord)

	eos = FactoredWord(EOS_TOKEN)
	eos.addFactor(LANGAUGE_FACTOR, UNK_TOKEN)
	eos.addFactor(SOUNDEX_FACTOR, UNK_TOKEN)
	res.append(eos)
	return res

def unkify(toUnk, mc, langMc, soundexMc):
	res = []
	for token in toUnk:
		if (mc.getRank(token.getWord()) > VOCAB_SIZE):
			token.setWord(UNK_TOKEN)
		if (langMc.getRank(token.getFactor(LANGAUGE_FACTOR)) > LANG_ID_VOCAB_SIZE):
			token.addFactor(LANGAUGE_FACTOR, UNK_TOKEN)
		if (soundexMc.getRank(token.getFactor(SOUNDEX_FACTOR)) > SOUNDEX_VOCAB_SIZE):
			token.addFactor(SOUNDEX_FACTOR, UNK_TOKEN)
		res.append(token)
	return res

f = open(INPUT_FILENAME, "r")
lines = f.readlines()
f.close()
print "Read file"

f = open(SOUNDEX_INPUT_FILE, "r")
soundexLines = f.readlines()
f.close()
print "Read file"

f = open(LANG_INPUT_FILE, "r")
langLines = f.readlines()
f.close()
print "Read file"

#CLEAN AND TOKENIZE THE LINES
cleanLines = []
for line, langLine, soundLine in zip(lines, langLines, soundexLines):
	cleanedLine      = cleanLine(line)
	cleanedLangLine  = cleanLine(langLine)
	cleanedSoundLine = cleanLine(soundLine)
	cleanLines.append(tokenize(cleanedLine, cleanedLangLine, cleanedSoundLine))
print "cleaned lines"

#GET THE COUNTS FROM THE SENTENCES
mc = Counts()
mcLang = Counts()
mcSoundex = Counts()
for line in cleanLines:
	for token in line:
		mc.incrementWord(token.getWord())
		mcLang.incrementWord(token.getFactor(LANGAUGE_FACTOR))
		mcSoundex.incrementWord(token.getFactor(SOUNDEX_FACTOR))
mc.fixRanks()
mcLang.fixRanks()
mcSoundex.fixRanks()
print "Counted the words"
print "Number of words in the dataset: ", len(mc)
print "Number of languages in the dataset: ", len(mcLang)
print "Number of sounds in the dataset: ", len(mcSoundex)

#REMOVE ALL OF THE UNKNOWN WORDS
unkedLines = []
for i, line in enumerate(cleanLines):
	unkedLines.append(unkify(line, mc, mcLang, mcSoundex))
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


