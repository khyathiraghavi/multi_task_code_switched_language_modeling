import os
import torch


#This is a simple dictionary that you can use to get an index for words and get the words from an index, should be fairly straight forward
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}									#This is the dict that stores the indexes of the words, keyed by the words themselves
        self.idx2word = []									#This is the reverse that gets the word based on the index, the index is the list index

    #This function adds a word to the dictionary
    #word - the word that is to be added
    #returns a numerical index for the word
    ## Add the word to the index list at the end and put the index number as the value for the word in the dictionary. Return the index.
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
    def __init__(self, path):
        self.dictionary = Dictionary()									#Define a dictionary for us to hold our words
        #self.train = self.tokenize(os.path.join(path, 'train.txt'))		#get all of the training examples
        self.train = self.tokenize(os.path.join(path, 'train_dummy.txt'))		#get all of the training examples
        self.valid = self.tokenize(os.path.join(path, 'valid_dummy.txt'))		#All of the validation examples
        self.test  = self.tokenize(os.path.join(path, 'test_dummy.txt'))		#and all of the testing examples

    #This function takes in a file path, reads it and then tokenizes the contents of each line in that file. The return value is a tensor (vector) that contains all the ids for the tokens in the file
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)										#Make sure the file exists
        # Add words to the dictionary
        with open(path, 'r') as f:										#Open the file 
            tokens = 0													#init a token counter
            for line in f:												#for every line in the file
                words = line.split() + ['<eos>']						#Split the line based on spaces and add an end of sequence token at the end
                tokens += len(words)									#Increment the token count 
                for word in words:										#Add all the words to the dictionary
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:										#open the file
            ids = torch.LongTensor(tokens)	#64 bit signed integer tensor	#Define a tensor to store all the ids that we get from the file
            token = 0													#define a token id
            for line in f:												#For every line in the file
                words = line.split() + ['<eos>']						#Split on spaces and add an end of sequence token
                for word in words:										#For every word in the sequence
                    ids[token] = self.dictionary.word2idx[word]			#lookup the id for that token
                    token += 1											#Then increment the token count
	#for k,v in self.dictionary.word2idx.iteritems():
	#	print k,v
	#print ids
	#raw_input()
        return ids 														#Finally return the tensor that contains the IDs

	# WHY IS TEST VOCAB BEING INCLUDED IN THE DICTIONARY?
	# WHY IS CREATING THE TENSOR IN A SEPARATE LOOP? COMPUTATIONALLY EXPENSIVE?
