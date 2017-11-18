import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import regex as re
import numpy as np
import csv
import copy

def readFbDataset():
	csvf = csv.reader(open('dataset-fb-valence-arousal-anon.csv'))
	messages = []
	def separateMessageFromScore(row):
		# print row
		# global messages
		messages.append(row[0])
		return row[1:]
	
	columns = np.array(next(csvf)[1:])
	scores = np.array([separateMessageFromScore(i) for i in csvf]).astype('int')
	messages = np.array(messages)

	return messages,columns,scores

# this doesn't parse the \t well
def readEmoBank():
	# csvf = csv.reader(open('emobank/raw.tsv'), delimiter='\t')
	csvf = []
	with open('emobank/raw.tsv') as eb:
		# csvf = csv.reader(open('emobank/raw.tsv'), delimiter='\t')
		line = eb.readline()
		while line:
			csvf.append(line.strip().split('\t'))
			line = eb.readline()
	msgdict = {}
	# next(csvf)
	# msgdict = { i[0].strip():i[1].strip() for i in csvf}
	del csvf[0]
	msgdict = { i[0]:i[1] for i in csvf }
	print msgdict
	print len(msgdict.keys())
	# asd
	# csvf = csv.reader(open('emobank/reader.tsv'), delimiter='\t')
	with open('emobank/reader.tsv') as eb:
		csvf = []
		line = eb.readline()
		while line:
			csvf.append(line.strip().split('\t'))
			line = eb.readline()

		messages = []
		def separateMessageFromScore(row):
			# print row
			# global messages
			messages.append(msgdict[row[0]])
			return row[1:]
		# asd
		columns = np.array(csvf[0][1:])
		del csvf[0]
		scores = np.array([separateMessageFromScore(i) for i in csvf]).astype('float')
		messages = np.array(messages)

	return messages,columns,scores
	
# print readEmoBank()


# csvf = csv.reader(open('dataset-fb-valence-arousal-anon.csv'))
# messages = np.array([])
# def separateMessageFromScore(row):
# 	# print row
# 	global messages
# 	messages = np.append(messages, row[0])
# 	return row[1:]

# columns = np.array(next(csvf)[1:])

# scores = np.array([separateMessageFromScore(i) for i in csvf]).astype('int')

# messages,columns,scores = readFbDataset()
messages,columns,scores = readEmoBank()

# print x1 == messages, x2 == columns, x3 == scores

words=[]

oldMessages = messages

messagesLimit = 2820
for i in range(len(messages[:messagesLimit])):
	sentence = messages[i].lower() 
	wordsInSentence = re.findall(r'\w+', sentence) 

	filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]
	words = words + filtered_words
	
	messages[i] = " ".join(filtered_words)

words, counts = np.unique(words, return_counts=True)
words = np.array([i for i,j in dict(zip(words, counts)).iteritems() if j > 1])
# words = [i for i,j in dict(zip(words, counts))]

# words = np.unique(np.array(words))
# print words
# print words.shape
# asd

def processSentence(s):
	
	sentence = s.lower() 
	wordsInSentence = re.findall(r'\w+', sentence) 

	filtered_words = [word for word in wordsInSentence if word not in stopwords.words('english')]

	s = filtered_words

	vector = np.array([0]*len(words))
	for i in s:
		index = np.where(words==i)[0]
		if len(index):
			vector[index[0]]+=1

	return vector


X=[]
for i in messages[:messagesLimit]:
	X.append(processSentence(i))
X = np.array(X).astype('float64')

Y=np.array(scores[:,1][:messagesLimit]).astype('float64')

def normaliseScores(scores):
	mm = np.mean(scores)
	mstd = np.std(scores)
	scores = (scores - mm) / mstd
	return scores

Y = normaliseScores(Y)
print Y.min()
print Y.max()
# plt.bar(np.arange(len(Y)),Y)
plt.bar(np.arange(len(Y)),sorted(Y))
# plt.bar(np.arange(len(Y)),[abs(i) for i in sorted(Y)])
plt.show()

def normaliseMatrix(a):
	mm = np.mean(a,axis=0)
	mstd = np.std(a,axis=0)
	a = (a - mm) / mstd
	
	return a

def meanAndStdDev(X,Y):
	# for i in X[:5]:
	# 	print "Mean of some X: ", np.mean(i) 
	# 	print "Std Dev of some X: ", np.std(i)	
	for i in range(5):
		print "Mean of some X: ", np.mean(X[:,i]) 
		print "Sum of some X: ", np.sum(X[:,i]) 
		print "Std Dev of some X: ", np.std(X[:,i])	

	print "Mean of Y: ", np.mean(Y) 
	print "Sum of Y: ", np.sum(Y) 
	print "Std Dev of Y: ", np.std(Y)

# X = X.astype('float64')


ss = np.array([sum(X[:,i]) for i in range(X.shape[1])])

# unique, counts = np.unique(ss, return_counts=True)
# print dict(zip(unique, counts))
X = normaliseMatrix(X)



meanAndStdDev(X,Y)

def train_model(X,Y):
	split = int(0.7*messagesLimit)
	x_train = X[:split]
	y_train = Y[:split]
	x_test = X[split:]
	y_test = Y[split:]

	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	input_layer_size = len(words)
	print "INPUT LAYER SIZE: %s"%input_layer_size
	model.add(Dense(256, activation='relu', input_dim=input_layer_size))
	model.add(Dropout(0.5))
	# model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(500, activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(100, activation='relu'))
	# model.add(Dropout(0.5))

	model.add(Dense(1, activation='linear'))
	# model.add(Dense(10, activation='softmax'))

	sgd = SGD(lr=0.0004, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error',
	# model.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['mae'])

	# check if it splits the train data as well
	model.fit(x_train, y_train,
			epochs=50,
			batch_size=512)
	score = model.evaluate(x_test, y_test, batch_size=32)
	print score
	# print y_test
train_model(X,Y)


# figure out if this model is useful
	# plot the data - distribution
	# play around interactively with it
	# put these results in a page
# reseach how to generate text 
	# basic model
	# nn model as well