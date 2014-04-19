#! /usr/bin/env python

import re
import string
import nltk

from bs4 import BeautifulSoup

f = open('word_stats.csv', 'a')

def main():
	count = 0
	print "Starting..."
	print
	#Read in the document
	print "Reading in the document..."
	print
	soup = BeautifulSoup(open("train_data.sgm"))
	print "Loaded train_data.sgm"
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
	
	#Get the list of stories
	articles = soup.findAll('story')
	
	print "Reading through individual stories now..."
	for art in articles:
		#Get the topic
		topic = art.find('topics')
		topic = topic.get_text()
		
		
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
		
		body = art.find('body')
		if body:
			body = body.get_text()
			body = body.split()
			#print body
		
			#Put each word of the article into the general hashmap
			for word in body:
				#print "word: " + word
				try:
					count = generalHashMap[word]
					count = count + 1
					generalHashMap[word] = count
				except KeyError:
					generalHashMap[word] = 1
		
			#For each of the topics, if they appear add the word to their specific hashmap
			if earnTopic:
				for word in body:
					try:
						count = earnHashMap[word]
						count = count + 1
						earnHashMap[word] = count
					except KeyError:
						earnHashMap[word] = 1
			if acqTopic:
				for word in body:
					try:
						count = acqHashMap[word]
						count = count + 1
						acqHashMap[word] = count
					except KeyError:
						acqHashMap[word] = 1
			if moneyTopic:
				for word in body:
					try:
						count = moneyHashMap[word]
						count = count + 1
						moneyHashMap[word] = count
					except KeyError:
						moneyHashMap[word] = 1
			if grainTopic:
				for word in body:
					try:
						count = grainHashMap[word]
						count = count + 1
						grainHashMap[word] = count
					except KeyError:
						grainHashMap[word] = 1
			if crudeTopic:
				for word in body:
					try:
						count = crudeHashMap[word]
						count = count + 1
						crudeHashMap[word] = count
					except KeyError:
						crudeHashMap[word] = 1
			if tradeTopic:
				for word in body:
					try:
						count = tradeHashMap[word]
						count = count + 1
						tradeHashMap[word] = count
					except KeyError:
						tradeHashMap[word] = 1
			if intTopic:
				for word in body:
					try:
						count = intHashMap[word]
						count = count + 1
						intHashMap[word] = count
					except KeyError:
						intHashMap[word] = 1
			if shipTopic:
				for word in body:
					try:
						count = shipHashMap[word]
						count = count + 1
						shipHashMap[word] = count
					except KeyError:
						shipHashMap[word] = 1
			if wheatTopic:
				for word in body:
					try:
						count = wheatHashMap[word]
						count = count + 1
						wheatHashMap[word] = count
					except KeyError:
						wheatHashMap[word] = 1	
			if cornTopic:
				for word in body:
					try:
						count = cornHashMap[word]
						count = count + 1
						cornHashMap[word] = count
					except KeyError:
						cornHashMap[word] = 1			
					
	#Remove words that only appear 3 or fewer times
	print
	print "Analysed stories"
	print
	print "Removing rare words..."
	print
	removeWords = []
	removed = 0
	for key in generalHashMap:
		if generalHashMap[key]<4:
			removeWords.append(key)
			
	for key in removeWords:
		generalHashMap.pop(key)
		earnHashMap.pop(key, "")
		acqHashMap.pop(key, "")
		moneyHashMap.pop(key, "")
		grainHashMap.pop(key, "")
		crudeHashMap.pop(key, "")
		tradeHashMap.pop(key, "")
		intHashMap.pop(key, "")
		shipHashMap.pop(key, "")
		wheatHashMap.pop(key, "")
		cornHashMap.pop(key, "")
		removed = removed + 1
	
	print
	print
	
	print "General: "
	print len(generalHashMap)
	print "Earn:"
	print len(earnHashMap)
	print "Acq:"
	print len(acqHashMap)
	print "Money:"
	print len(moneyHashMap)
	print "Grain:"
	print len(grainHashMap)
	print "Crude:"
	print len(crudeHashMap)
	print "Trade:"
	print len(tradeHashMap)
	print "Interest:"
	print len(intHashMap)
	print "Ship:"
	print len(shipHashMap)
	print "Wheat:"
	print len(wheatHashMap)
	print "Corn:"
	print len(cornHashMap)
	print
	print "Words removed:"
	print removed
	
	print
	print "Creating csv outputs..."
	titleOutput = "word,general,earn,acq,money,grain,crude,trade,interest,ship,wheat,corn\n"
	output_to_file(titleOutput)
	for word in generalHashMap:
		general = generalHashMap[word]
		try: 
			earnHashMap[word]
			earn = earnHashMap[word]
		except KeyError:
			earn=0
		try:
			acqHashMap[word]
			acq = acqHashMap[word]
		except KeyError:
			acq = 0
		try:
			moneyHashMap[word]
			money = moneyHashMap[word]
		except KeyError:
			money = 0
		try:
			grainHashMap[word]
			grain = grainHashMap[word]
		except KeyError:
			grain = 0
		try:
			crudeHashMap[word]
			crude = crudeHashMap[word]
		except KeyError:
			crude = 0
		try:
			tradeHashMap[word]
			trade = tradeHashMap[word]
		except KeyError:
			trade = 0
		try:
			intHashMap[word]
			int = intHashMap[word]
		except KeyError:
			int = 0
		try:
			shipHashMap[word]
			ship = shipHashMap[word]
		except KeyError:
			ship = 0
		try:
			wheatHashMap[word]
			wheat = wheatHashMap[word]
		except KeyError:
			wheat = 0
		try:
			cornHashMap[word]
			corn = cornHashMap[word]
		except KeyError:
			corn = 0
		outputLine = word + "," + str(general) + "," + str(earn) + "," + str(acq) + "," + str(money) + "," + str(grain) + "," + str(crude) + "," + str(trade) + "," + str(int) + "," + str(ship) + "," + str(wheat) + "," + str(corn) + "\n"
		output_to_file(outputLine)
		
	print 
	print "Done"
	
def output_to_file(output):
	global f
	#print output
	try:
		f.write(output)
	except UnicodeEncodeError:
		print "HAHA, YOU'RE NOT STOPPING ME THIS TIME ERROR, FUCK YOU"
	
main()