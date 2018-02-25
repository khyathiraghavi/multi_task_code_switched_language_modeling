import nltk

from Counts import Counts
from FactoredWord import FactoredWord
from random import shuffle

############################################### FILE CONSTANTS ###############################################

INPUT_FILENAME = "data/dedup_sentences.txt"
SOUNDEX_INPUT_FILE = "data/dedup_soundex.txt"
LANG_INPUT_FILE = "data/dedup_lang_ids.txt"

OUTPUT_FILENAME = "output/output.txt"

############################################### DATA CONSTANTS ###############################################

VOCAB_SIZE = 5000
LANG_ID_VOCAB_SIZE = 5
SOUNDEX_VOCAB_SIZE = 2000
MAX_SENTENCE_LEN = 100
MAKE_UPPER = True

TRAIN_PERCENT = 0.8
TEST_PERCENT = 0.1
VALIDATION_PERCENT = 0.1

############################################### TOKEN CONSTANTS ###############################################

UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

SOUNDEX_UNK_TOKEN = "<UNK_SX>"
SOUNDEX_EOS_TOKEN = "<EOS_SX>"

LANG_OTHER_TOKEN = "O"

WORD_FACTOR = "WORD_FACTOR"
LANGAUGE_FACTOR = "LANG_FACTOR"
SOUNDEX_FACTOR = "SOUNDEX_FACTOR"

FACTORS_LIST = [WORD_FACTOR, LANGAUGE_FACTOR, SOUNDEX_FACTOR]

############################################### CODE BELOW ###############################################

def cleanLine(toClean):
	res = toClean.replace("\n", "").replace("\r", "").replace("\t", " ").lstrip(" ").rstrip(" ")
	if(MAKE_UPPER):
		res = res.upper()
	return res

def factorize(toTokenize, lanIDsToTokenize, soundexToTokenize):
	splits        =        toTokenize.split(" ")
	soundexSplits = soundexToTokenize.split(" ")
	langSplits    =  lanIDsToTokenize.split(" ")

	toks        =        splits[0:min(MAX_SENTENCE_LEN, len(splits))]
	langToks    =    langSplits[0:min(MAX_SENTENCE_LEN, len(splits))]
	soundexToks = soundexSplits[0:min(MAX_SENTENCE_LEN, len(splits))]
	
	assert(len(toks) == len(soundexToks) and len(soundexToks) == len(langToks)) # Make sure that all the factors have the same lengths
	
	res = []
	for token, langToken, soundToken in zip(toks, langToks, soundexToks):
		fWord = FactoredWord(FACTORS_LIST)
		fWord.addFactor(WORD_FACTOR,     token)
		fWord.addFactor(LANGAUGE_FACTOR, langToken)
		fWord.addFactor(SOUNDEX_FACTOR,  soundToken)
		res.append(fWord)

	eos = FactoredWord(FACTORS_LIST)
	eos.addFactor(WORD_FACTOR,     EOS_TOKEN)
	eos.addFactor(LANGAUGE_FACTOR, LANG_OTHER_TOKEN)
	eos.addFactor(SOUNDEX_FACTOR,  SOUNDEX_EOS_TOKEN)
	res.append(eos)

	return res

def unkify(toUnk, mc, langMc, soundexMc):
	res = []
	for token in toUnk:
		if (mc.getRank(token.getFactor(WORD_FACTOR)) > VOCAB_SIZE):
			token.addFactor(WORD_FACTOR, UNK_TOKEN)
		if (langMc.getRank(token.getFactor(LANGAUGE_FACTOR)) > LANG_ID_VOCAB_SIZE):
			token.addFactor(LANGAUGE_FACTOR, LANG_OTHER_TOKEN)
		if (soundexMc.getRank(token.getFactor(SOUNDEX_FACTOR)) > SOUNDEX_VOCAB_SIZE):
			token.addFactor(SOUNDEX_FACTOR, SOUNDEX_UNK_TOKEN)
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

#CLEAN AND FACTORIZE THE LINES
cleanLines = []
for line, langLine, soundLine in zip(lines, langLines, soundexLines):
	cleanedLine      = cleanLine(line)
	cleanedLangLine  = cleanLine(langLine)
	cleanedSoundLine = cleanLine(soundLine)
	cleanLines.append(factorize(cleanedLine, cleanedLangLine, cleanedSoundLine))
print "cleaned lines"

#GET THE COUNTS FROM THE SENTENCES
mc = Counts()
mcLang = Counts()
mcSoundex = Counts()
for line in cleanLines:
	for token in line:
		mc.incrementWord(token.getFactor(WORD_FACTOR))
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


