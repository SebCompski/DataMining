#! /usr/bin/env python

import re
import string
import nltk
import pickle

from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup

id = 0
f = open('stage_one.sgm', 'a')

def main():
	print "Starting..."
	#Read in the document
	soup = BeautifulSoup(open("reut2-000.sgm"))
	process_file(soup)
	print "5%"
	soup = BeautifulSoup(open("reut2-001.sgm"))
	process_file(soup)
	print "10%"
	soup = BeautifulSoup(open("reut2-002.sgm"))
	process_file(soup)
	print "15%"
	soup = BeautifulSoup(open("reut2-003.sgm"))
	process_file(soup)
	print "20%"
	soup = BeautifulSoup(open("reut2-004.sgm"))
	process_file(soup)
	print "25%"
	soup = BeautifulSoup(open("reut2-005.sgm"))
	process_file(soup)
	print "30%"
	soup = BeautifulSoup(open("reut2-006.sgm"))
	process_file(soup)
	print "35%"
	soup = BeautifulSoup(open("reut2-007.sgm"))
	process_file(soup)
	print "40%"
	soup = BeautifulSoup(open("reut2-008.sgm"))
	process_file(soup)
	print "45%"
	soup = BeautifulSoup(open("reut2-009.sgm"))
	process_file(soup)
	print "50%"
	soup = BeautifulSoup(open("reut2-010.sgm"))
	process_file(soup)
	print "55%"
	soup = BeautifulSoup(open("reut2-011.sgm"))
	process_file(soup)
	print "60%"
	soup = BeautifulSoup(open("reut2-012.sgm"))
	process_file(soup)
	print "65%"
	soup = BeautifulSoup(open("reut2-013.sgm"))
	process_file(soup)
	print "70%"
	soup = BeautifulSoup(open("reut2-014.sgm"))
	process_file(soup)
	print "75%"
	soup = BeautifulSoup(open("reut2-015.sgm"))
	process_file(soup)
	print "80%"
	soup = BeautifulSoup(open("reut2-016.sgm"))
	process_file(soup)
	print "85%"
	soup = BeautifulSoup(open("reut2-017.sgm"))
	process_file(soup)
	print "90%"
	soup = BeautifulSoup(open("reut2-018.sgm"))
	process_file(soup)
	print "95%"
	soup = BeautifulSoup(open("reut2-019.sgm"))
	process_file(soup)
	print "97-ish%"
	soup = BeautifulSoup(open("reut2-020.sgm"))
	process_file(soup)
	print "99%?"
	soup = BeautifulSoup(open("reut2-021.sgm"))
	process_file(soup)
	print "100%!"

def output_to_file(output):
	global f
	#print output
	try:
		f.write(output)
	except UnicodeEncodeError:
		print "HAHA, YOU'RE NOT STOPPING ME THIS TIME ERROR, FUCK YOU"
	
def process_file(soup):
	output = ""
	
	#Final all instances of the unwanted tags
	dates = soup.findAll('date') 
	places = soup.findAll('places')
	people = soup.findAll('people')
	orgs = soup.findAll('orgs')
	exchanges = soup.findAll('exchanges')
	companies = soup.findAll('companies')
	unknowns = soup.findAll('unknown')

	#Remove all instances of the unwanted tags
	[line.extract() for line in dates]
	[line.extract() for line in places]
	[line.extract() for line in people]
	[line.extract() for line in orgs]
	[line.extract() for line in exchanges]
	[line.extract() for line in companies]
	[line.extract() for line in unknowns]

	#For all of the body tags, take their strings, do the pre-processing on them
	articles = soup.findAll('reuters')
	
	
	
	#For each article, run the process article method
	for art in articles:
		topic = art.find('topics')
		iTopic = topic.find_all('d')
		if iTopic:
			for t in iTopic:
				t = t.get_text()
				#print t
				if t == "earn" or t == "acq" or t == "money-fx" or t == "grain" or t == "crude" or t == "trade" or t == "interest" or t == "ship" or t == "wheat" or t == "corn":
					#print "This one is good"
					output = process_article(art)
					output_to_file(output)
					break
		
def process_article(articles):
	idoutput = "<ID>" + articles.get("newid") + "</ID>\n"
	lewisout = "<LEWISSPLIT>" + articles.get("lewissplit") + "</LEWISSPLIT>\n"
	
	#Process their topics
	topicoutput = ""
	topic = articles.find('topics')
	if topic:
		topicoutput = process_topic(topic)
		topicoutput = "<topics>" + topicoutput + "</topics>\n"

	#Process their titles
	titleoutput = ""
	title = articles.find('title')
	if title:
		titleoutput = process_punctuation(title)
		titleoutput = process_POS(titleoutput)
		titleoutput = process_lemm(titleoutput)
		titleoutput = process_removewords(titleoutput)
		titleoutput = "<title> " + titleoutput + "</title>\n"
		
	#Process their datelines
	datelineoutput = ""
	dateline = articles.find('dateline')
	if dateline:
		datelineoutput = process_punctuation(dateline)
		datelineoutput = process_POS(datelineoutput)
		datelineoutput = process_lemm(datelineoutput)
		datelineoutput = process_removewords(datelineoutput)
		datelineoutput = "<dateline> " + datelineoutput + "</dateline>\n"
		
	#Process their bodies
	bodyoutput = ""
	body = articles.find('body')
	if body:
		bodyoutput = process_punctuation(body)
		bodyoutput = process_POS(bodyoutput)
		bodyoutput = process_lemm(bodyoutput)
		bodyoutput = process_removewords(bodyoutput)
		bodyoutput = "<body> " + bodyoutput + "</body>\n"
		#print bodyoutput
		
	#print "<story>\n" + idoutput + lewisout + topicoutput + titleoutput + datelineoutput + bodyoutput + "</story>\n"
	return "<story>\n" + idoutput + lewisout + topicoutput + titleoutput + datelineoutput + bodyoutput + "</story>\n"

def process_topic(body):
	temp=""
	topics = body.find_all('d')
	outputtopic = ""
	for t in topics:
		#print t
		temp = t.get_text()
		outputtopic = outputtopic + temp + " "
	
	return outputtopic

def process_punctuation(body):
	newLine = body.get_text()

	newLine = re.sub('\.|,|<|>|\'|:|;|(|)|-|\*|/|\\|?|!|"|%|$|#', '', newLine)
	newLine = re.sub('Reuter', '', newLine)
	newLine = newLine.lower()
	#print "Punct: ", newLine
	return newLine
	
def process_removewords(newLine):
	newLine = re.sub(' the-DT | be-VB | to-TO | of-IN | and-CC | a-DT | in-IN| i-JJ | i-NN | i-PRP | i--NONE- | it-PRP | for-IN | on | is-VBZ | if-JJ | if-IN | at-IN | an-DT | or-CC | so-RB | so-IN', ' ', newLine)

	newLine = re.sub('[0-9]+', 'NUMBER', newLine)
	#print "Remove: ", newLine
	return newLine
	
def process_POS(newLine):
	newLine = newLine.split()
	newLine = nltk.pos_tag(newLine)
	outLine=""
	for tup in newLine:
		st = tup[0] + "-" + tup[1]
		outLine = outLine + " " + st
	#print "POS: ", outLine
	
	return outLine
	
def process_lemm(newLine):
	newLine = newLine.split()
	lmtzr = WordNetLemmatizer()
	for x in range(0,len(newLine)):
		newLine[x] = lmtzr.lemmatize(newLine[x])
		newLine[x] = lmtzr.lemmatize(newLine[x], "v")
	
	
	newLine = " ".join(newLine)
	#print newLine
	return newLine

main()