import re
import sys
import string
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
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
import json

#Default
given_emotion = 'anger'
number_of_generations = 3
filename = None
print(sys.argv)
if(len(sys.argv)>1):
    given_emotion = sys.argv[1]

if(len(sys.argv)>2):
    number_of_generations = int(sys.argv[2])

if(len(sys.argv)>3):
    filename = sys.argv[3]



print("Emotion: " + given_emotion)
print("Number of output sequences: ", number_of_generations)
print("Filename: " + str(filename))
#df = pd.read_csv("brexit/brexit_labelled_tweets_" + given_emotion + ".csv", sep=';')
df = pd.read_csv("brexit/attempt3/brexit_labelled_tweets_" + given_emotion + ".csv", sep=';')
emotion_filter = df['emotion'] == given_emotion
tweets = df[emotion_filter]['text']

tweets = tweets[:40000]
if(given_emotion == 'fear'):
    tweets = tweets[:24000]

print("Tweets: " + str(len(tweets)))


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

# Reshape dataX to size of [samples, time steps, features] and scale it to 0-1
# Represent dataY as one hot encoding

#num_classes = max(map(max, dataX))
num_classes = max( max(map(max, dataX)), max(dataY) ) + 1


def generate_output(model, start=None):
    if(start is None):
        start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start][:]
    result = []
    input_seq = ' '.join([int_to_word[value] for value in pattern])
    #print("Seed:")
    #print("\"", ' '.join([int_to_word[value] for value in pattern]), "\"")
    for i in range(200):
        #x = np.reshape(pattern, (1, len(pattern), 1))
        #x = x/float(n_vocab)
        x = np.array([np_utils.to_categorical(pattern, num_classes=num_classes)])
        prediction = model.predict(x)
        #index = np.argmax(prediction)
        index = np.random.choice( num_classes, 1 , p = prediction[0])[0]
        result.append(int_to_word[index])
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    #print("\nGenerated Sequence:")
    #print(' '.join(result))
    #print("\nDone.")
    return (input_seq, ' '.join(result))


model = load_model("data/best_models/"+ given_emotion + ".hdf5")

def generate_evaluation_data(model, number_of_generations, filename=None):
    eval_data = {'human':[], 'computer':[]}
    generated_t = []
    for i in range(number_of_generations//2):
        input_seq, output_seq = generate_output(model)
        size = np.random.randint(12, 25)
        output_seq = " ".join(output_seq.split()[:size])
        generated_t.append([input_seq,output_seq])
        eval_data['computer'].append({"input":input_seq, "output":output_seq})
    real_t = []
    for i in range(number_of_generations//2):
        start = np.random.randint(0, len(dataX)-1)
        pattern = dataX[start][:]
        input_seq = ' '.join([int_to_word[value] for value in pattern])
        size = np.random.randint(12, 25)
        result = []
        for i in range(size):
             result.append(int_to_word[dataX[start + i + 1][-1]])
        result = ' '.join(result)
        real_t.append([input_seq, result])
        eval_data['human'].append({"input":input_seq, "output":result})
    #if(not filename is None):
    with open('evaluation/'+filename+'.json', 'w') as outfile:
        json.dump(eval_data, outfile)    
    return (generated_t, real_t, eval_data)

if __name__ == "__main__":
    if(filename is None):
        for i in range(number_of_generations):
            input_seq, output_seq = generate_output(model)
            print("INPUT: ",input_seq)
            print("OUTPUT: ", output_seq)
    else:
         generate_evaluation_data(model, number_of_generations, filename)





