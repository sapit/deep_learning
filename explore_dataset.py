import pandas as pd
from main import *
from keras.models import Sequential

# df = pd.read_csv('election-day-tweets/election_day_tweets.csv')
# df = pd.read_csv('SemEval2018/EI-reg-En-anger-train.txt', sep='\t')
df = pd.read_csv('SemEval2018/EI-reg-En-joy-train.txt', sep='\t')


# modelname="data/weights/emotion-detection-weights-improvement-48-0.7389.hdf5"
modelname="data/weights/emotion-detection-weights-improvement-47-0.6804.hdf5"

model = Sequential()
input_layer_size = len(words)
print "INPUT LAYER SIZE: %s"%input_layer_size
model.add(Dense(256, activation='relu', input_dim=input_layer_size))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

sgd = SGD(lr=0.0004, decay=1e-6, momentum=0.9, nesterov=True)
            
model.load_weights(modelname)
model.compile(loss='mean_squared_error',
            optimizer=sgd,
            metrics=['mae'])

# print df["text"][0]

def preprocessSentence(s):
    new_s=[]
    for w in s.split():
        if(w.startswith('@')):
            continue
        if(w.startswith('#')):
            w.replace('#','')
        new_s.append(w)
    return " ".join(new_s)

def vector_to_sentence(v):
    return " ".join([idx_to_word[i] for i in range(len(v)) if v[i]>0])

# def main():
# offset = 10000
offset = 0
predictions = []
word_loss_list = []
for i in range(1600):
    # org_text = df["text"][i]
    org_text = df["Tweet"][i]
    text = preprocessSentence(org_text)
    vector = processSentence(text)
    prediction = model.predict(np.array(vector).reshape(1,len(vector)))
    # print ""
    # print prediction[0][0]
    predictions.append(prediction[0][0])
    if(len(text.split()) == 0):
        continue
    word_loss = float(len(vector_to_sentence(vector).split())) / len(text.split())
    word_loss_list.append(word_loss)
    if 5>6:
        print ""
        print org_text
        print vector_to_sentence(vector)
        print model.predict(np.array(vector).reshape(1,len(vector)))
        print ""

predictions = np.array(predictions)
word_loss_list = np.array(word_loss_list)


def classifyList(l):
    # 0.1 .. 0.9
    classes = [float(i)/10 for i in range(1,10,1)]
    classes = zip(classes, classes[1:])
    res = [filter(lambda x:x>=i and x<j ,l) for i,j in classes]
    sizes = map(len, res)
    return sizes
