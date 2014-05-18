#! /usr/bin/env python

import re
import string
import nltk


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import Ward
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import KFold
from bs4 import BeautifulSoup

#f = open('word_stats.csv', 'a')
f = open('results.sgm', 'w')
s = open('stats.sgm', 'w')

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
		if topic:
			topic = topic.get_text()
			#print t

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

			#Get the body and place it into the array at the current index point.
			body = art.find('body')
			if body:
				body = body.get_text()
			else:
				body = "0"
			split = art.find('lewissplit')

			#for each topic, if it is on, add it to the topic array, you don't want to add ones that are in there but aren't one of the important 10
			if earnTopic:
				classArray.append("earn")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif acqTopic:
				classArray.append("acq")
				storyArray.append(body)
				splitArray.append(split)
			elif moneyTopic:
				classArray.append("money")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif grainTopic:
				classArray.append("grain")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif crudeTopic:
				classArray.append("crude")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif tradeTopic:
				classArray.append("trade")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif intTopic:
				classArray.append("interest")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif shipTopic:
				classArray.append("ship")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif wheatTopic:
				classArray.append("wheat")
				storyArray.append(body)
				split = split.get_text()
				splitArray.append(split)
			elif cornTopic:
				classArray.append("corn")
				storyArray.append(body)
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
	
	print "Naive Bayes:"
	gnb = GaussianNB()
	print "Doing k-fold cross fold validation stuff..."
	currentFold = 1
	bestAccuracy = 0.0
	kf = KFold(len(testClass), 10)
	for trainkf, testkf in kf:
		print ("Doing fold: %d" % currentFold)
		#trainkf and testkf are a list of indicies, so you need to pull them out
		splitTrainStory = [trainStory[i] for i in trainkf]
		splitTrainClass = [trainClass[i] for i in trainkf]

		splitTestStory = [trainStory[i] for i in testkf]
		splitTestClass = [trainClass[i] for i in testkf]

		#Make the model and do the important naive bayesy stuff
		model = gnb.fit(splitTrainStory, splitTrainClass)
		labels = model.classes_
		prediction = model.predict(splitTestStory)
		output_csv(splitTestClass, prediction,labels,currentFold,"Naive Bayes")
		output_stats(splitTestClass,prediction,labels,currentFold,"Naive Bayes")
		#print_metrics(splitTestClass, prediction)

		if (bestAccuracy < float(metrics.accuracy_score(splitTestClass, prediction))):
			bestAccuracy = float(metrics.accuracy_score(splitTestClass, prediction))
			bestModel = model
			bestPrediction = prediction
			bestTestClass = splitTestClass

		print ("This model accuracy: %0.3f" % float(metrics.accuracy_score(splitTestClass, prediction)))
		print ("Current best Acc: %0.3f" % bestAccuracy)

		currentFold = currentFold + 1

	print
	print ("Best NB Accuracy: %0.3f" % bestAccuracy)
	print "Running against the test data now..."
	prediction = model.predict(testStory)
	print_metrics(testClass, prediction)
	output_csv(testClass, prediction,labels,"Best","Naive Bayes")
	output_stats(testClass,prediction,labels,"Best","Naive Bayes")
	print


	print "SVM:"
	sup = svm.LinearSVC()
	print "Doing k-fold cross fold validation stuff..."
	currentFold = 1
	bestAccuracy = 0.0
	kf = KFold(len(testClass), 10)
	for trainkf, testkf in kf:
		print ("Doing fold: %d" % currentFold)
		#trainkf and testkf are a list of indicies, so you need to pull them out
		splitTrainStory = [trainStory[i] for i in trainkf]
		splitTrainClass = [trainClass[i] for i in trainkf]

		splitTestStory = [trainStory[i] for i in testkf]
		splitTestClass = [trainClass[i] for i in testkf]

		model = sup.fit(trainStory, trainClass)
		labels = model.classes_
		prediction = model.predict(splitTestStory)
		output_csv(splitTestClass, prediction,labels,currentFold,"SVM")
		output_stats(splitTestClass,prediction,labels,currentFold,"SVM")
		#print_metrics(testClass, prediction)

		if (bestAccuracy < float(metrics.accuracy_score(splitTestClass, prediction))):
			bestAccuracy = float(metrics.accuracy_score(splitTestClass, prediction))
			bestModel = model
			bestPrediction = prediction
			bestTestClass = splitTestClass

		print ("This model accuracy: %0.3f" % float(metrics.accuracy_score(splitTestClass, prediction)))
		print ("Current best Acc: %0.3f" % bestAccuracy)

		currentFold = currentFold + 1

	print
	print ("Best SVM Accuracy: %0.3f" % bestAccuracy)
	print "Running against the test data now..."
	prediction = model.predict(testStory)
	print_metrics(testClass, prediction)
	output_csv(testClass, prediction,labels,"Best","SVM")
	output_stats(testClass,prediction,labels,"Best","SVM")
	print

	print "Random Forest:"
	rand = RandomForestClassifier(n_estimators=10)
	print "Doing k-fold cross fold validation stuff..."
	currentFold = 1
	bestAccuracy = 0.0
	kf = KFold(len(testClass), 10)
	for trainkf, testkf in kf:
		print ("Doing fold: %d" % currentFold)
		#trainkf and testkf are a list of indicies, so you need to pull them out
		splitTrainStory = [trainStory[i] for i in trainkf]
		splitTrainClass = [trainClass[i] for i in trainkf]

		splitTestStory = [trainStory[i] for i in testkf]
		splitTestClass = [trainClass[i] for i in testkf]

		model = rand.fit(trainStory, trainClass)
		labels = model.classes_
		prediction = model.predict(splitTestStory)
		output_csv(splitTestClass, prediction,labels,currentFold,"RAndom Forest")
		output_stats(splitTestClass,prediction,labels,currentFold,"Random Forest")
		#print_metrics(testClass, prediction)

		if (bestAccuracy < float(metrics.accuracy_score(splitTestClass, prediction))):
			bestAccuracy = float(metrics.accuracy_score(splitTestClass, prediction))
			bestModel = model
			bestPrediction = prediction
			bestTestClass = splitTestClass

		print ("This model accuracy: %0.3f" % float(metrics.accuracy_score(splitTestClass, prediction)))
		print ("Current best Acc: %0.3f" % bestAccuracy)

		currentFold = currentFold + 1

	print
	print ("Best RF Accuracy: %0.3f" % bestAccuracy)
	print "Running against the test data now..."
	prediction = model.predict(testStory)
	print_metrics(testClass, prediction)
	output_csv(testClass, prediction,labels,"Best","Random Forest")
	output_stats(testClass,prediction,labels,"Best","Random Forest")
	print

	clusterTrainStory = []
	clusterTrainClass = []
	clusterTestStory = []
	clusterTestClass = []

	print
	print "Splitting stories and stuff for clustering..."

	for art in articles:
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
			classArray.append(topic)

		split = art.find('lewissplit')
		split = split.get_text()
		splitArray.append(split)


	for x in range(0,len(storyArray)-1):
		if (splitArray[x]=="TRAIN"):
			clusterTrainStory.append(storyArray[x])
			clusterTrainClass.append(classArray[x])
		elif (splitArray[x]=="TEST"):
			clusterTestStory.append(storyArray[x])
			clusterTestClass.append(classArray[x])

	#clusterTrainStory = StandardScaler().fit_transform(clusterTrainStory)
	lsa = TruncatedSVD()
	newStoryArray = vectorizer.fit_transform(clusterTrainStory)
	newStoryArray = lsa.fit_transform(newStoryArray)
	newStoryArray = normalize(newStoryArray)

	print
	print "DBSCAN:"
	classifier = DBSCAN(eps=0.3, min_samples=10)
	db = classifier.fit(newStoryArray)
	labels = db.labels_
	print_cluster(clusterTrainClass, labels, newStoryArray)

	print
	print "KMeans:"
	classifier = KMeans(n_clusters=81)
	db = classifier.fit(newStoryArray)
	labels = db.labels_
	print_cluster(clusterTrainClass, labels, newStoryArray)

	print
	print "Ward:"
	classifier = Ward(n_clusters=81,copy=True)
	db = classifier.fit(newStoryArray)
	labels = db.labels_
	print_cluster(clusterTrainClass, labels, newStoryArray)

	print 
	print "Finished"
	print "Exiting.."


def print_cluster(clusterTrainClass, labels, clusterTestStory):
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(clusterTrainClass, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(clusterTrainClass, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(clusterTrainClass, labels))
	print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(clusterTrainClass, labels))
	print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(clusterTrainClass, labels))
	print "Silhouette Coefficient:"
	print metrics.silhouette_score(clusterTestStory, labels, metric='euclidean')
	
def output_csv(testClass,prediction,labels,fold,model):
	print "Writing confusion matrix..."
	cm = confusion_matrix(testClass, prediction,labels)
	global f

	columnTotals = [0,0,0,0,0,0,0,0,0,0]
	rowTotals = [0,0,0,0,0,0,0,0,0,0]

	f.write("\n")
	f.write(model)
	f.write("Fold: " + str(fold) + "\n")
	labelString = ','.join(labels)
	labelString = "," + labelString + "\n"
	f.write(labelString)
	for x in range(0,len(labels)-1):
		labelOut = str(labels[x])
		f.write(labelOut)
		for y in range(0,len(labels)-1):
			cmOut = "," + str(cm[x][y])
			rowTotals[x] = rowTotals[x] + cm[x][y]
			columnTotals[y] = columnTotals[y] + cm[x][y]
			f.write(cmOut)
		rowFigure = "," + str(rowTotals[x])
		f.write(rowFigure)
		f.write("\n")
	columnFigure = ','.join("'{0}'".format(n) for n in columnTotals)
	columnFigure = "Totals," + columnFigure
	columnFigure = re.sub('\'', '', columnFigure)
	f.write(columnFigure)
	f.write("\n")
	f.write("\n")

def output_stats(testClass,prediction,labels,fold,model):
	print "Writing stats..."
	global s

	s.write(model + "\n")
	s.write(str(fold) + "\n")
	s.write(classification_report(testClass,prediction, target_names=labels))
	
	#print(classification_report[1])

	s.write("\n")


def print_metrics(testClass, prediction):
	print "Accuracy:"
	print metrics.accuracy_score(testClass, prediction)
	print "Precision, macro then micro:"
	print metrics.precision_score(testClass, prediction, average='macro')
	print metrics.precision_score(testClass, prediction, average='micro')
	print "Recall, macro then micro:"
	print metrics.recall_score(testClass, prediction, average='macro')
	print metrics.recall_score(testClass, prediction, average='micro')
	print "F1 score, macro then micro:"
	print metrics.f1_score(testClass, prediction, average='macro')
	print metrics.f1_score(testClass, prediction, average='micro')
	print

main()