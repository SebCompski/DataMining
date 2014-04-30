#! /usr/bin/env python

import re
import string
import nltk


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from bs4 import BeautifulSoup

#f = open('word_stats.csv', 'a')

def main():
	count = 0
	print "Starting..."
	print
	#Read in the document
	print "Reading in the document..."
	print
	soup = BeautifulSoup(open("stage_one.sgm"))
	print "Loaded stage_one.sgm"
	print
	
	#Create Hash Maps
	print "Creating Hash Maps..."
	generalHashMap = {}
	earnHashMap = {}
	acqHashMap = {}
	moneyHashMap = {}
	grainHashMap = {}
	crudeHashMap = {}
	tradeHashMap = {}
	intHashMap = {}
	shipHashMap = {}
	wheatHashMap = {}
	cornHashMap = {}

	storyArray = []
	classArray = []
	splitArray = [] #True means its for training, false means its for testing
	
	#Get the list of stories
	articles = soup.findAll('story')
	
	print "Reading through individual stories now..."
	for art in articles:
		#Get the topic
		topic = art.find('topics')
		topic = topic.get_text()

		#Get the body and place it into the array at the current index point.
		body = art.find('body')
		if body:
			body = body.get_text()
			storyArray.append(body)
		else:
			storyArray.append('0')
		
		#for each doc, find which topics are in it
		earnTopic = re.search("earn", topic)
		acqTopic = re.search("acq", topic)
		moneyTopic = re.search("money", topic)
		grainTopic = re.search("grain", topic)
		crudeTopic = re.search("crude", topic)
		tradeTopic = re.search("trade", topic)
		intTopic = re.search("interest", topic)
		shipTopic = re.search("ship", topic)
		wheatTopic = re.search("wheat", topic)
		cornTopic = re.search("corn", topic)

		#for each topic, if it is on, add it to the topic array, you don't want to add ones that are in there but aren't one of the important 10
		if earnTopic:
			classArray.append("earn")
		elif acqTopic:
			classArray.append("acq")
		elif moneyTopic:
			classArray.append("money")
		elif grainTopic:
			classArray.append("grain")
		elif crudeTopic:
			classArray.append("crude")
		elif tradeTopic:
			classArray.append("trade")
		elif intTopic:
			classArray.append("interest")
		elif shipTopic:
			classArray.append("ship")
		elif wheatTopic:
			classArray.append("wheat")
		elif cornTopic:
			classArray.append("corn")
		else:
			classArray.append("ERROR")

		split = art.find('lewissplit')
		split = split.get_text()
		splitArray.append(split)

	print
	print "Starting split and vectorisation..."
	print

	print "Class array size: "
	print len(classArray)
	print "Story array size: "
	print len(storyArray)

	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
	sparse = vectorizer.fit_transform(storyArray)
	sparse = sparse.toarray()

	print 
	print "Vectorised..."
	print "Splitting now..."
	print

	#classes = vectorizer.fit_transform(classArray)
	#classes = classes.toarray()

	trainStory = []
	trainClass = []
	testStory = []
	testClass = []

	for x in range(0,len(splitArray)-1):
		if (splitArray[x]=="TRAIN"):
			trainStory.append(sparse[x])
			trainClass.append(classArray[x])
		elif (splitArray[x]=="TEST"):
			testStory.append(sparse[x])
			testClass.append(classArray[x])

	print
	print "Split stats:"
	print ("Train stories: %d" % len(trainStory))
	print ("Test Stories: %d" % len(testStory))
	print
	print "Running model creator..."
	
	'''
	print "Naive Bayes:"
	gnb = GaussianNB()
	model = gnb.fit(trainStory, trainClass)
	gnbpred = model.fit(trainStory, trainClass)
	print gnbpred
	print ("Number of mislabeled points: %d" % (testClass != gnbpred).sum())
	print metrics.classification_report(testClass, gnbpred)
	print
	'''

	print "SVM:"
	sup = svm.LinearSVC()
	model = sup.fit(trainStory, trainClass)
	prediction = model.predict(testStory)
	print ("Number of mislabeled points: %d" % (testClass != prediction).sum())
	print metrics.classification_report(testClass, prediction)

	print "Random Forest:"
	rand = RandomForestClassifier(n_estimators=10)
	model = rand.fit(trainStory, trainClass)
	prediction = model.predict(testStory)
	print ("Number of mislabeled points: %d" % (testClass != prediction).sum())
	print metrics.classification_report(testClass, prediction)


	print 
	print "Finished"
	print "Exiting.."
	
main()