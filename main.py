import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import regex as re
import numpy as np
import csv
import copy
import read_dataset as rd
import model

def process_tweet(s):
    new_s=[]
    for w in s.split():
        if(w.startswith('@')):
            continue
        if(w.startswith('#')):
            w.replace('#','')
        new_s.append(w)
    return " ".join(new_s)    

def vectoriseSentence(s):
	sentence = s.lower()
	wordsInSentence = re.findall(r'\w+', sentence)
	filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]
	s = filtered_words
	
	vector = np.array([0]*len(words))
	for i in s:
		index = np.where(words==i)[0]
		if len(index):
			vector[index[0]]+=1
			# vector[index[0]]=1
	return vector

def normaliseScores(scores):
	mm = np.mean(scores)
	mstd = np.std(scores)
	scores = (scores - mm) / mstd
	return scores

def normaliseMatrix(a):
	mm = np.mean(a,axis=0)
	mstd = np.std(a,axis=0)
	a = (a - mm) / mstd
	return a

def processMessage(m, words):
	sentence = m.lower() 
	wordsInSentence = re.findall(r'\w+', sentence) 
	filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]
	words.extend(filtered_words)
	return " ".join(filtered_words)

messages,columns,scores = rd.readEmoBank()

words=[]
messagesLimit = 2820

for i in range(len(messages[:messagesLimit])):
	sentence = messages[i].lower() 
	wordsInSentence = re.findall(r'\w+', sentence) 
	filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]
	words = words + filtered_words
	messages[i] = " ".join(filtered_words)
# messages = map(lambda x: processMessage(x,words), messages[:messagesLimit])
# print messages

words, counts = np.unique(words, return_counts=True)
idx_to_word = np.array([i for i,j in dict(zip(words, counts)).iteritems() if j > 1])
word_to_idx = {idx_to_word[i]:i for i in range(len(idx_to_word))}
words = idx_to_word



X = [vectoriseSentence(i) for i in messages[:messagesLimit] ]
X = np.array(X).astype('float64')

Y = np.array(scores[:,1][:messagesLimit]).astype('float64')
# plt.hist(Y)
# plt.show()

X = normaliseMatrix(X)
Y = normaliseScores(Y)



if __name__ == "__main__":
	model.train_model(X,Y)


# figure out if this model is useful
	# plot the data - distribution
	# play around interactively with it
	# put these results in a page
# reseach how to generate text 
	# basic model
	# nn model as well