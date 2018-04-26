import random

TRAIN = 0.6
TEST = 0.2
VAL = 0.2


f = open("langs.txt", "r")
langLines = f.readlines()
f.close()

f = open("words.txt", "r")
wordLines = f.readlines()
f.close()

combined = zip(wordLines, langLines)

random.shuffle(combined)

trainOffset = 0
testOffset = int(len(combined)*TRAIN)
valOffset = int(len(combined)*TRAIN) + int(len(combined)*TEST)
train = combined[trainOffset:testOffset]
test = combined[testOffset:valOffset]
val = combined[valOffset:]

f = open("wtrain.txt", "w")
f2 = open("ltrain.txt", "w")
for words, langs in train:
	f.write(words + "\n")
	f2.write(langs + "\n")
f.close()
f2.close()

f = open("wtest.txt", "w")
f2 = open("ltest.txt", "w")
for words, langs in test:
	f.write(words + "\n")
	f2.write(langs + "\n")
f.close()
f2.close()

f = open("wvalid.txt", "w")
f2 = open("lvalid.txt", "w")
for words, langs in val:
	f.write(words + "\n")
	f2.write(langs + "\n")
f.close()
f2.close()