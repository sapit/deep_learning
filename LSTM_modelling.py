import re
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from main import process_tweet
from read_dataset import readSemEval2018joy

# rawtext = open('wonderland.txt','r').read().split('\n')
rawtext = readSemEval2018joy()
rawtext = ' '.join(rawtext)
rawtext = [word.strip(string.punctuation) for word in rawtext.split()]
rawtext = ' '.join(rawtext)
rawtext = rawtext.replace('-', ' ')
rawtext = ' '.join(rawtext.split())

all_words = rawtext.split()
unique_words = sorted(list(set(all_words)))
n_vocab = len(unique_words)
print "Total Vocab:", n_vocab
word_to_int = dict((w, i) for i, w in enumerate(unique_words))
int_to_word = dict((i, w) for i, w in enumerate(unique_words))

raw_text = rawtext.split()
n_words = len(raw_text)
print n_words

seq_length = 100
dataX = []
dataY = []
for i in xrange(0, n_words - seq_length):
    seq_in  = raw_text[i: i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([word_to_int[word] for word in seq_in])
    dataY.append(word_to_int[seq_out])
n_patterns = len(dataX)
print 'Total patterns:', n_patterns

# Reshape dataX to size of [samples, time steps, features] and scale it to 0-1
# Represent dataY as one hot encoding
X_train = np.reshape(dataX, (n_patterns, seq_length, 1))/float(n_vocab)
Y_train = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print model.summary()

# define the checkpoint
filepath="data/weights/word-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, nb_epoch=10, batch_size=128, callbacks=callbacks_list)

# load the network weights
# filename = "data/weights/weights-improvement-18-2.8349.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.fit(X_train, Y_train, nb_epoch=80, batch_size=32, callbacks=callbacks_list)

def trainMore(model):
    model.fit(X_train, Y_train, nb_epoch=80, batch_size=32, callbacks=callbacks_list)

# load the network weights
# filename = "weights-improvement-18-2.8349.hdf5"
# model.load_weights(filename)
# model.compile(loss='categorical_crossentropy', optimizer='adam')

start = np.random.randint(0, len(X_train)-1)
pattern = dataX[start]
result = []
print "Seed:"
print "\"", ' '.join([int_to_word[value] for value in pattern]), "\""
for i in xrange(200):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x/float(n_vocab)
    prediction = model.predict(x)
    index = np.argmax(prediction)
    result.append(int_to_word[index])
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print "\nGenerated Sequence:"
print ' '.join(result)
print "\nDone."  

