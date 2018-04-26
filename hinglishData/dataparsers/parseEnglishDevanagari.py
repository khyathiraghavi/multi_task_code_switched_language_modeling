import io
from nltk.tokenize.moses import MosesTokenizer

INPUT_PATH = "../spellNormalized_Full.txt"
OUTPUT_L_PATH = "langs.txt"
OUTPUT_W_PATH = "words.txt"

DEVNAGRI_MIN = int("0900", 16)
DEVNAGRI_MAX = int("097F", 16)

ENGLISH = 0
DEVNAGRI = 1
OTHER = 2 

TOKENIZER = MosesTokenizer()

def tokenize(line):
	return TOKENIZER.tokenize(line)
	#return line.split(" ")

def isDevnagri(char):
	charVal = ord(char)
	if(charVal < DEVNAGRI_MIN or charVal > DEVNAGRI_MAX):
		return False
	return True

def isEnglish(char):
	return char.isalpha()

def isOther(char):
	return not char.isalpha()

def identifyWordLangage(word):
	eVote = 0
	dVote = 0
	oVote = 0
	for char in word:
		if(isDevnagri(char)):
			dVote += 1
		elif(isEnglish(char)):
			eVote += 1
		else:
			oVote += 1
	votes = [eVote, dVote, oVote]
	if(max(votes) == eVote):
		return ENGLISH
	elif(max(votes) == dVote):
		return DEVNAGRI
	return OTHER

f = io.open(INPUT_PATH, mode="r", encoding="utf-8")
lines = f.readlines()
f.close()
wordRes = ""
langRes = ""
for line in lines:
	words = tokenize(line)
	langs = []
	good = False
	for word in words:
		langId = identifyWordLangage(word)
		langs.append(unicode(str(langId), "utf-8"))
		if(langId == DEVNAGRI):
			good = True
	if(good):
		wordRes += " ".join(words) + "\n"
		langRes += " ".join(langs) + "\n"

f = io.open(OUTPUT_L_PATH, mode="w", encoding="utf-8")
f.write(langRes)
f.close()

f = io.open(OUTPUT_W_PATH, mode="w", encoding="utf-8")
f.write(wordRes)
f.close()