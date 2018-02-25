INPUT_FILENAME = "data/sentences.txt"
SOUNDEX_INPUT_FILE = "data/soundex.txt"
LANG_INPUT_FILE = "data/lang_ids.txt"

OUTPUT_FILENAME = "output/dedup_sentences.txt"
SOUNDEX_OUTPUT_FILE = "output/dedup_soundex.txt"
LANG_OUTPUT_FILE = "output/dedup_lang_ids.txt"

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


res = []
soundexRes = []
langRes = []


for line, soundex, lang in zip(lines, soundexLines, langLines):
	if not line in res:
		res.append(line)
		soundexRes.append(soundex)
		langRes.append(lang)

f = open(OUTPUT_FILENAME, "w")
f.write("".join(res))
f.close()
print "Read file"

f = open(SOUNDEX_OUTPUT_FILE, "w")
f.write("".join(soundexRes))
f.close()
print "Read file"

f = open(LANG_OUTPUT_FILE, "w")
f.write("".join(langRes))
f.close()
print "Read file"