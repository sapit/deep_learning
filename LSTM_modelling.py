import re
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
import pandas as pd
from utils import *
import copy
import pickle
import keras.backend as K
import pickle
import sys

#df = pd.read_csv("brexit_labelled_tweets.csv")
#emotion_filter = df['emotion'] == 'anger'
#emotion_filter = df['emotion'] == 'joy'
#emotion_filter = df['emotion'] == 'sadness'
#tweets = list(filter(lambda s: eval_english(s)<0.7 , map(process_tweet, df[emotion_filter]['tweet'])))
#print("Tweets: " + str(len(tweets)))

#from main import process_tweet
#from read_dataset import readSemEval2018joy

given_emotion = 'anger'
#given_emotion = 'fear'

if(len(sys.argv)>1):
    given_emotion = sys.argv[1]

print("Emotion: " + given_emotion)

#tweets = []
#for i in range(1,11):
#    print("Processing Batch " + str(i))
#    df = pd.read_csv("brexit/brexit_labelled_tweets_batch_" + str(i) + "_en.csv", sep=';')
#    #emotion_filter = df['emotion'] == 'anger'
#    emotion_filter = df['emotion'] == given_emotion
#    tweets.extend(df[emotion_filter]['text'])

#df=None

#given_emotion = 'fear'

df = pd.read_csv("brexit/attempt3/brexit_labelled_tweets_" + given_emotion + ".csv", sep=';')
emotion_filter = df['emotion'] == given_emotion
tweets = df[emotion_filter]['text']
all_tweets = tweets

if __name__=="__main__":
    tweets = tweets[:40000]
    if(given_emotion == 'fear'):
        tweets = tweets[:24000]

tweets = tweets[:4000]
#tweets = [t + " <END>" for t in tweets]
print("Tweets: " + str(len(tweets)))
print(tweets[1])

rawtext = open('wonderland.txt','r').read().split('\n')
#rawtext = tweets
rawtext = rawtext[:len(rawtext)]
rawtext.extend(tweets[:len(tweets)])
#rawtext = readSemEval2018joy()
rawtext = ' '.join(rawtext)
rawtext = [word.strip(string.punctuation) for word in rawtext.split()]
rawtext = ' '.join(rawtext)
rawtext = rawtext.replace('-', ' ')
rawtext = ' '.join(rawtext.split())
rawtext = [word.strip(string.punctuation) for word in rawtext.split()]
rawtext = ' '.join(rawtext)

all_words_initial = rawtext.split()
all_words = []
token = "<TOKEN>"
infrequent_words = set()
token_added = False
unique_words, counts = np.unique(all_words_initial, return_counts=True)
for i in range(len(unique_words)):
    if counts[i]>10:
        all_words.append(unique_words[i])
    else:
        if(not token_added):
            all_words.append(token)
            token_added = True
        infrequent_words.add(unique_words[i])
rawtext = rawtext.split()
for i in range(len(rawtext)):
    if(rawtext[i] in infrequent_words):
        rawtext[i] = token
rawtext = ' '.join(rawtext)


#all_words = rawtext.split()
unique_words = sorted(list(set(all_words)))
n_vocab = len(unique_words)
print("Total Vocab:", n_vocab)
word_to_int = dict((w, i) for i, w in enumerate(unique_words))
int_to_word = dict((i, w) for i, w in enumerate(unique_words))

raw_text = rawtext.split()
n_words = len(raw_text)
print(n_words)

seq_length = 15
#seq_length = 50
dataX = []
dataY = []
for i in range(0, n_words - seq_length):
    seq_in  = raw_text[i: i+seq_length]
    seq_out = raw_text[i+seq_length]

    if("<TOKEN>" in seq_in or "<TOKEN>" in seq_out):
        continue

    dataX.append([word_to_int[word] for word in seq_in])
    dataY.append(word_to_int[seq_out])
n_patterns = len(dataX)
print('Total patterns:', n_patterns)

print("max: ",max(dataY))
print("max: ",max(map(max,dataX)))

# Reshape dataX to size of [samples, time steps, features] and scale it to 0-1
# Represent dataY as one hot encoding

num_classes = max( max(map(max, dataX)), max(dataY) ) + 1

p = np.random.permutation(len(dataX))
dataX = np.array(dataX)
dataY = np.array(dataY)
dataX = dataX[p]
dataY = dataY[p]

#X = np.array(list(map(lambda a: np_utils.to_categorical(a, num_classes = num_classes) , dataX)))
#X = np.array( [np_utils.to_categorical(d, num_classes = num_classes) for d in dataX] )



#X = np_utils.to_categorical(dataX, num_classes = num_classes)
#Y = np_utils.to_categorical(dataY)
#X_train = X
#Y_train = Y

if __name__ == "__main__" or True:
    X_train = np_utils.to_categorical(dataX, num_classes = num_classes)
    Y_train = np_utils.to_categorical(dataY, num_classes = num_classes)


#p = np.random.permutation(len(X))
#X_train = X[p]
#Y_train = Y[p]

#adam = Adam(lr=0.004, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model = Sequential()
#model.add(GRU(256, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(GRU(512, return_sequences=True, input_shape=(dataX.shape[1], num_classes)))
model.add(Dropout(0.2))
model.add(GRU(512))
model.add(Dropout(0.2))

#model.add(GRU(1024, return_sequences=True, input_shape=(dataX.shape[1], num_classes)))
#model.add(Dropout(0.2))
#model.add(GRU(512, return_sequences=True))
#model.add(Dropout(0.2))
#model.add(GRU(1024))
#model.add(Dropout(0.2))
#model.add(GRU(512, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(GRU(512))

#model.add(Dropout(0.55))
#model.add(Dense(Y_train.shape[1], activation='softmax'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam)
print(model.summary())


#adam2 = Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adam2 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
mmodel = Sequential()
mmodel.add(GRU(1024, input_shape=(dataX.shape[1], num_classes)))
#mmodel.add(Dropout(0.2))
mmodel.add(Dropout(0.5))
mmodel.add(Dense(num_classes, activation='softmax'))
mmodel.compile(loss='categorical_crossentropy', optimizer=adam2)
print(mmodel.summary())


# define the checkpoint
filepath="data/two_layer_gru_" + given_emotion + "/weights/word-weights-improvement-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5"
#filepath="data/" + given_emotion + "/weights/word-weights-improvement-{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

def myGenerator(dataX, dataY, num_classes=None, b_size=30000):
    if num_classes is None:
        num_classes = max(dataY) + 1
    while 1:
        for i in range(len(dataX)//b_size): 
            print("Processing batch: ",i+1)
            yield np_utils.to_categorical(dataX[i*b_size:(i+1)*b_size], num_classes = num_classes), np_utils.to_categorical(dataY[i*b_size:(i+1)*b_size], num_classes = num_classes)

def train_with_generator(super_epochs=10, batch_size=20000, lr_decrease=2):
    val_split = int(0.05 * len(dataX))
    valX, valY = np_utils.to_categorical(dataX[:val_split], num_classes = num_classes), np_utils.to_categorical(dataY[:val_split], num_classes=num_classes)
    trainX, trainY = dataX[val_split:], dataY[val_split:]
    #batch_size = 30000
    g = myGenerator(trainX, trainY , num_classes = num_classes, b_size = batch_size)
    val_loss = []
    loss = []
    history=[]
    print("Number of batches: ", len(trainX)//batch_size) 
    for j in range(super_epochs):
        print("Super epoch: ",j+1)
        for i in range(len(trainX)//batch_size):
            X_train, Y_train = next(g)
            #h = mmodel.fit(X_train, Y_train, nb_epoch=1, batch_size=4096, validation_data=(valX, valY), callbacks=callbacks_list)
            h = mmodel.fit(X_train, Y_train, nb_epoch=1, batch_size=4096, validation_data=(valX, valY))
            X_train, Y_train = None, None
            #history.append(h)
            val_loss += h.history['val_loss']
            loss += h.history['loss']
        lr = K.get_value(mmodel.optimizer.lr)
        K.set_value(mmodel.optimizer.lr, lr/lr_decrease)
    return loss,val_loss


if  __name__=="__main__":

    history = model.fit(X_train, Y_train, nb_epoch=80, batch_size=4096, validation_split=0.1, callbacks=callbacks_list)
    
    #history = model.fit(X_train, Y_train, nb_epoch=100, batch_size=4096, validation_split=0.1, callbacks=callbacks_list)
    #K.set_value(adam.lr, 0.0003)
    #history2 = model.fit(X_train, Y_train, nb_epoch=400, batch_size=4096, validation_split=0.1, callbacks=callbacks_list)
    

    
    #history = mmodel.fit(X_train, Y_train, nb_epoch=5, batch_size=4096, validation_split=0.1, callbacks=callbacks_list)
    #K.set_value(adam2.lr, 0.0007)
    #history2 = mmodel.fit(X_train, Y_train, nb_epoch=50, batch_size=4096, validation_split=0.1, callbacks=callbacks_list)

    
    with open("data/two_layer_gru_" + given_emotion + "/history.pck", 'wb') as f:
    #with open("data/sequence_50/" + given_emotion + "/history.pck", 'wb') as f:
        pickle.dump(history.history, f)
        #f.close()

    with open("data/two_layer_gru_" + given_emotion + "/history2.pck", 'wb') as f:
    #with open("data/sequence_50/" + given_emotion + "/history2.pck", 'wb') as f:
        #pickle.dump(history2.history, f)
        pass
        #f.close()

    #size = len(X_train)
    #val_x, X_train = X_train[:size//20], X_train[size//20:]
    #val_y, Y_train = Y_train[:size//20], Y_train[size//20:]
    
    #model.fit(X_train, Y_train, nb_epoch=200, batch_size=1024, validation_data=(val_x, val_y), callbacks=callbacks_list)
    #K.set_value(adam.lr, 0.0003)
    #model.fit(X_train, Y_train, nb_epoch=200, batch_size=1024, validation_data=(val_x, val_y), callbacks=callbacks_list)
    #K.set_value(adam.lr, 0.00006)
    #model.fit(X_train, Y_train, nb_epoch=200, batch_size=1024, validation_data=(val_x, val_y), callbacks=callbacks_list)
    


    #K.set_value(adam.lr, 0.00005)
    #model.fit(X_train, Y_train, nb_epoch=100, batch_size=64, validation_split=0.1, callbacks=callbacks_list)
    #K.set_value(adam.lr, 0.00001)
    #model.fit(X_train, Y_train, nb_epoch=100, batch_size=64, validation_split=0.1, callbacks=callbacks_list)

    #model.fit(X_train, Y_train, nb_epoch=200, batch_size=64, callbacks=callbacks_list)
    #K.set_value(adam.lr, 0.00002)
    #model.fit(X_train, Y_train, nb_epoch=1000, batch_size=64, callbacks=callbacks_list)
    #K.set_value(adam.lr, 0.000006)
    #model.fit(X_train, Y_train, nb_epoch=1000, batch_size=64, callbacks=callbacks_list)

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

def generate_output(model, start=None):
    if(start is None):
        start = np.random.randint(0, len(X_train)-1)
    #pattern = dataX[start][:]
    pattern = dataX[start].tolist()
    result = []
    print("Seed:")
    print("\"", ' '.join([int_to_word[value] for value in pattern]), "\"")
    for i in range(200):
        #x = np.reshape(pattern, (1, len(pattern), 1))
        #x = x/float(n_vocab)
        x = np.array([np_utils.to_categorical(pattern, num_classes=num_classes)])
        prediction = model.predict(x)
        #index = np.argmax(prediction)
        index = np.random.choice( n_vocab, 1 , p = prediction[0])[0]
        result.append(int_to_word[index])
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nGenerated Sequence:")
    print(' '.join(result))
    print("\nDone.")
    return ' '.join(result)


if __name__=="__main__":
    generate_output(model)
