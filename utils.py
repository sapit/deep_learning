import numpy as np
import regex as re
from nltk.corpus import stopwords

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

def vectoriseSentence(s, words):
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

def predictions_from_raw(ts, model, words):
	predictions=[]
	vectors = []
	for t in ts:
		t=process_tweet(t)
		t=processMessage(t,[])
		vs=vectoriseSentence(t, words)
		vectors.append(vs)
	predictions=model.predict(np.array(vectors))
	# predictions.append((t,p))
	# print(t)
	# print(p)
	return predictions

def vector_to_emotion(v,emotions=None):
	if emotions is None:
		emotions = ["joy",
			"sadness",
			"anger",
			"fear"]
	v=list(v)
	emotion = emotions[v.index(max(v))]
	return emotion


