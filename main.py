import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# print x_train
# print x_train.shape
# print y_train[0]
# print y_train.shape

import numpy as np
import csv
# data = np.genfromtxt(("\t".join(i) for i in csv.reader(open('dataset-fb-valence-arousal-anon.csv'))), delimiter="\t")

# data = np.genfromtxt('dataset-fb-valence-arousal-anon.csv', dtype=None, delimiter=',', names=True)

csv = csv.reader(open('dataset-fb-valence-arousal-anon.csv'))

messages = np.array([])
def separateMessageFromScore(row):
	global messages
	# print row[0]
	messages = np.append(messages, row[0])
	return row[1:]

columns = np.array(next(csv)[1:])

scores = np.array([separateMessageFromScore(i) for i in csv]).astype('int')
	

# print messages
# print columns
# print scores



# for i in range(len(columns)):
# 	print columns[i]
# 	print scores[:,i]


words=[]

oldMessages = messages

messagesLimit = 2820

for i in messages[:messagesLimit]:
	for j in i.strip().split():
		words.append(j)
words = np.unique(np.array(words))

# words = words[100:12000]
# print words
# print words.shape

# words = np.array([j for j in i for i in messages])

# print words
# print words.shape


def processSentence(s):
	s = s.strip().split()
	vector = np.array([0]*len(words))
	for i in s:
		index = np.where(words==i)[0]
		if index:
			vector[index[0]]+=1

	return vector



# X=np.array([])
X=[]
for i in messages[:messagesLimit]:
	X.append(processSentence(i))
	# messages = np.append(messages, row[0])
X = np.array(X)

# print X
# print X.shape
# print scores[:,0]
# print scores[:,0].shape
# print messages.shape
Y=scores[:,1][:messagesLimit]

def normaliseScores(scores):
	newScores = []
	for i in scores:
		score = np.array([0]*10)
		score[i]=1
		newScores.append(score)
	return np.array(newScores)

Y = normaliseScores(Y)

# x_train = np.random.random((1000, 20))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
# x_test = np.random.random((100, 20))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# print x_train.shape
# print x_train
# print y_train.shape
# print y_train


split = int(0.7*messagesLimit)
x_train = X[:split]
y_train = Y[:split]
x_test = X[split:]
y_test = Y[split:]

# print x_train.shape
# print y_train.shape

# print "LEN ", len(words) 

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