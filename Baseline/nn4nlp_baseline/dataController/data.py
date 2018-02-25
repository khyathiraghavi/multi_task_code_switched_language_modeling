import os
import torch

#This is a simple dictionary that you can use to get an index for words and get the words from an index, should be fairly straight forward
class FactorDictionary(object):
    def __init__(self):
        self.word2idx = {}									#This is the dict that stores the indexes of the words, keyed by the words themselves
        self.idx2word = []									#This is the reverse that gets the word based on the index, the index is the list index

    #This function adds a word to the dictionary
    #word - the word that is to be added
    #returns a numerical index for the word
    def add_word(self, word):								
        if word not in self.word2idx:						#If we dont already have an index for the word, then we need to add it
            self.idx2word.append(word)						#Add the word to the list of words
            self.word2idx[word] = len(self.idx2word) - 1	#The index of that word then becomes the length of the word list -1 because it is now at the end
        return self.word2idx[word]							#Then we look up the word, and then return its index

    #This function gets the vocabulary size
    def __len__(self):	
        return len(self.idx2word)							#Return the number of indexes that we have have, which is the vocabulary size


#This is a simple class that stores all of our training, test, and validation data
class Corpus(object):
	#This is the constructor, it takes a folder path to the train, test, and validation data
    def __init__(self, path, factorsList):
        self.dictionaries = {}
        for factorLabel in factorsList:
            self.dictionaries[factorLabel] = FactorDictionary()

        self.dictify(path + '.train')		#get all of the training examples
        self.dictify(path + '.val')		#All of the validation examples
        self.dictify(path + '.test')		    #and all of the testing examples

    def getFactorDict(self, factorName):
        return self.dictionaries[factorName]

    #This function takes in a file path, reads it and then tokenizes the contents of each line in that file. The return value is a tensor (vector) that contains all the ids for the tokens in the file
    def dictify(self, path):
        # Add words to the dictionary
        with open(path, 'r') as f:										#Open the file 
            tokens = 0													#init a token counter
            for line in f:												#for every line in the file
                words = line.split()
                tokens += len(words)									#Increment the token count 
                for word in words:										#Add all the words to the dictionary
                    for factor in word.split("/"):
                        label, value = factor.split(":")
                        self.dictionaries[label].add_word(value)

    def convertToData(self, path, factorsInfo):
        with open(path, 'r') as f:                                      #Open the file 
            tokens = 0                                                  #init a token counter
            for line in f:                                              #for every line in the file
                words = line.split()
                tokens += len(words)

        factorCount = len(self.dictionaries.keys())
        # Tokenize file content
        with open(path, 'r') as f:										#open the file
            ids = torch.LongTensor(tokens, factorCount)         		#Define a tensor to store all the ids that we get from the file
            token = 0													#define a token id
            for line in f:												#For every line in the file
                words = line.split()
                for word in words:                                      #Add all the words to the dictionary
                    for factor in word.split("/"):
                        label, value = factor.split(":")
                        fidx = factorsInfo.getFactorOrder().index(label)
                        ids[token][fidx] = self.dictionaries[label].word2idx[value]
                    token += 1
        return ids 														#Finally return the tensor that contains the IDs
