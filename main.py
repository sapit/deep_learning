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
import re

def process_tweet(s):
	url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
	# map(re.findall(pattern,s))
	# reduce()
	for url in re.findall(url_pattern,s):
		s = s.replace(url,"")
	new_s=[]
	for w in s.split():
		if(w.startswith('@')):
			continue
		if(w.startswith('#')):
			w = w.replace('#','')
		new_s.append(w)
	return " ".join(new_s)    

def vectoriseSentence(s):
	s = s.split()
	vector = np.array([0]*len(words))
	for i in s:
		index = np.where(words==i)[0]
		if len(index):
			# vector[index[0]]+=1
			vector[index[0]]=1
	return vector
def vectorToWords(a):
	idxs = np.where(a>=1)[0]
	return list(map(idx_to_word.item, idxs))

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

def processMessage(m, r_words=[]):
	sentence = m.lower() 
	wordsInSentence = re.findall(r'\w+', sentence) 
	filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]
	r_words.extend(filtered_words)
	# r_words = np.concatenate((r_words, filtered_words))
	return " ".join(filtered_words)

def fun(ts):
	predictions=[]
	for t in ts:
		t=process_tweet(t)
		t=processMessage(t,[])
		vs=vectoriseSentence(t)
		p=mymodel.predict(np.array([vs]))
		predictions.append((t,p))
		print(t)
		print(p)
	return predictions

if __name__ == "__main__":
	messages,columns,scores = rd.readEmoBank()

	words=[]
	# messagesLimit = 2820

	for i in range(len(messages)):
		sentence = messages[i].lower() 
		wordsInSentence = re.findall(r'\w+', sentence) 
		filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]
		words = words + filtered_words
		messages[i] = " ".join(filtered_words)
	# messages = map(lambda x: processMessage(x,words), messages)
	# print messages

	words, counts = np.unique(words, return_counts=True)
	idx_to_word = np.array([i for i,j in dict(zip(words, counts)).items() if j > 1])
	word_to_idx = {idx_to_word[i]:i for i in range(len(idx_to_word))}
	words = idx_to_word


	X = [vectoriseSentence(i) for i in messages ]
	X = np.array(X).astype('float64')

	Y = np.array(scores[:,1]).astype('float64')
	# plt.hist(Y)
	# plt.show()

	X = normaliseMatrix(X)
	Y = normaliseScores(Y)



if __name__ == "__main__":
	mymodel = model.train_model(X,Y)


# figure out if this model is useful
	# plot the data - distribution
	# play around interactively with it
	# put these results in a page
# reseach how to generate text 
	# basic model
	# nn model as well