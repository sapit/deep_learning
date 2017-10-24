import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import csv

csv = csv.reader(open('dataset-fb-valence-arousal-anon.csv'))

messages = np.array([])
def separateMessageFromScore(row):
	global messages
	messages = np.append(messages, row[0])
	return row[1:]

columns = np.array(next(csv)[1:])

scores = np.array([separateMessageFromScore(i) for i in csv]).astype('int')
	
words=[]

oldMessages = messages

messagesLimit = 2820

for i in messages[:messagesLimit]:
	for j in i.strip().split():
		words.append(j)
words = np.unique(np.array(words))

def processSentence(s):
	s = s.strip().split()
	vector = np.array([0]*len(words))
	for i in s:
		index = np.where(words==i)[0]
		if index:
			vector[index[0]]+=1

	return vector


X=[]
for i in messages[:messagesLimit]:
	X.append(processSentence(i))
X = np.array(X)

Y=scores[:,1][:messagesLimit]

def normaliseScores(scores):
	newScores = []
	for i in scores:
		score = np.array([0]*10)
		score[i]=1
		newScores.append(score)
	return np.array(newScores)

Y = normaliseScores(Y)

split = int(0.7*messagesLimit)
x_train = X[:split]
y_train = Y[:split]
x_test = X[split:]
y_test = Y[split:]

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(7000, activation='relu', input_dim=len(words)))
# model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=128)
score = model.evaluate(x_test[:832], y_test[:832], batch_size=32)
print score