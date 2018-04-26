from gensim.models.word2vec import Word2Vec

LANG_FILE = "langs.txt"
#LANG_FILE = "../hinglishData/dataparsers/langs.txt"

f = open(LANG_FILE, "r")
sentences = [line.split(" ") for line in f.readlines()]
f.close()

w2vModel = Word2Vec(sentences, size=10, window=5, min_count=5, workers=4)

ENGLISH = 0
DEVNAGRI = 1
OTHER = 2 
line = ""
f = open("langVecs.txt", "w")
line +=  "ENGLISH " + " ".join([str(val) for val in w2vModel.wv["0"]]) + "\n"
line +=  "DEVNAGRI " + " ".join([str(val) for val in w2vModel.wv["1"]]) + "\n"
line +=  "OTHER " + " ".join([str(val) for val in w2vModel.wv["2"]]) + "\n"
f.write(line)
f.close()