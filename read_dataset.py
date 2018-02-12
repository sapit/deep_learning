import numpy as np
import csv
import copy
import pandas as pd

def readSemEval2018(e="joy"):
	emotions = ["joy", "anger", "fear", "sadness"]
	if e not in emotions:
		return None
	# EI-reg-En-joy-train.txt
	df = pd.read_csv('./SemEval2018/EI-reg-En-%s-train.txt'%(e), sep='\t')
	return df["Tweet"]

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
	# csvf = csv.reader(open('/scratch/2144328i/raw.tsv'), delimiter='\t')
	csvf = []
	with open('/scratch/2144328i/raw.tsv') as eb:
		# csvf = csv.reader(open('/scratch/2144328i/raw.tsv'), delimiter='\t')
		line = eb.readline()
		while line:
			csvf.append(line.strip().split('\t'))
			line = eb.readline()
	msgdict = {}
	# next(csvf)
	# msgdict = { i[0].strip():i[1].strip() for i in csvf}
	del csvf[0]
	msgdict = { i[0]:i[1] for i in csvf }
	# print msgdict
	# print len(msgdict.keys())
	# asd
	# csvf = csv.reader(open('/scratch/2144328i/reader.tsv'), delimiter='\t')
	with open('/scratch/2144328i/reader.tsv') as eb:
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

def readSmileDataset():
	df = pd.read_csv('smile/smile-annotations-final.csv')
	return df["tweet"], df.columns, df["emotion"]

def readSmileDatasetDf():
	return pd.read_csv('smile/smile-annotations-final.csv')
