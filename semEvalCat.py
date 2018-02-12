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
from main import vectoriseSentence, normaliseScores, normaliseMatrix, processMessage, process_tweet, vectorToWords
import json

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def plot_precisions(model, x_test, y_test):
    correct = []
    incorrect = []
    for i in range(len(x_test)):
        pr = model.predict(np.array([x_test[i]]))[0]
        val = 1 - max(pr)
        # print(pr)
        # print(y_test[i])
        if np.where(pr==max(pr))[0][0] != np.where(y_test[i]==max(y_test[i]))[0][0]:
            incorrect.append((val,i))
        else:
            correct.append((val,i))
    a,b = zip(*correct)
    plt.plot(b,a, 'bo')
    a,b = zip(*incorrect)
    plt.plot(b,a, 'ro')
    plt.show()

# mymodel = model.create_categorical_model(X.shape[1])
# mymodel.load_weights("data/weights/emotion-detection-weights-improvement-14-0.2643.hdf5")
def eval_with_smile():
    onehot = {
        "happy": [1,0,0,0],
        "sad": [0,1,0,0],
        "angry":[0,0,1,0]
    }
    df = rd.readSmileDatasetDf()
    X=[]
    Y=[]
    for i,row in df.iterrows():
        pass
        if(row["emotion"] in onehot.keys()):
            vector = vectoriseSentence(processMessage(process_tweet(row["tweet"]),[]))
            X.append(vector)
            Y.append(onehot[row["emotion"]])
    return X,Y


	# x_happy = df.loc[df["emotion"]=="happy"]["tweet"].tolist()
	# x_sad = df.loc[df["emotion"]=="sad"]["tweet"].tolist()
	# x_angry = df.loc[df["emotion"]=="angry"]["tweet"].tolist()
	# y_happy = [[1,0,0,0]] * len(x_happy)
	# y_sad = [[1,0,0,0]] * len(x_sad)
	# y_angry = [[1,0,0,0]] * len(x_angry)

emotions = ["joy",
            "sadness",
            "anger",
            "fear"]

joy = rd.readSemEval2018("joy")
sadness = rd.readSemEval2018("sadness")
anger = rd.readSemEval2018("anger")
fear = rd.readSemEval2018("fear")
Y =     [emotions.index("joy")]*len(joy)\
      + [emotions.index("sadness")]*len(sadness)\
      + [emotions.index("anger")]*len(anger)\
      + [emotions.index("fear")]*len(fear)
print("Messages: %s"%len(Y))
words = []
messages = []
# joy + sadness + anger + fear
for e in emotions:
    messages.extend(rd.readSemEval2018(e))

messages = list(map(lambda x: processMessage(x,words), list(map(process_tweet,messages))))
words, counts = np.unique(words, return_counts=True)
idx_to_word = np.array([i for i,j in dict(zip(words, counts)).items() if j > 1])
word_to_idx = {idx_to_word[i]:i for i in range(len(idx_to_word))}
words = idx_to_word

with open("words_list.json", "w") as outfile:
    json.dump(list(words), outfile)

# X = [vectoriseSentence(i) for i in messages ]
X = list(map(vectoriseSentence, messages))
X = np.array(X).astype('float64')

# Y = np.array(scores[:,1]).astype('float64')
Y = convertToOneHot(np.array(Y))

temp = list(zip(X,Y))
np.random.shuffle(temp)
X,Y = zip(*temp)
X = np.array(X)
Y = np.array(Y)

# plt.hist(Y)
# plt.show()

# X = normaliseMatrix(X)
# Y = normaliseScores(Y)

if __name__ == "__main__":
	mymodel = model.train_categorical_model(X,Y)
