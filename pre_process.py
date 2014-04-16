#! /usr/bin/env python

import re
import string
import nltk
import pickle

from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup

id = 0
f = open('pre_processed_records.sgm', 'a')

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
		output = process_article(art)
		print output
		output_to_file(output)
		
def process_article(articles):
	global id
	idoutput = "<ID>" + str(id) + "</ID>\n"
	id = id + 1

	#Process their topics
	topicoutput = ""
	topic = articles.find('topics')
	if topic:
		topicoutput = process_punctuation(topic)
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
		
	return idoutput + topicoutput + titleoutput + datelineoutput + bodyoutput


def process_punctuation(body):
	newLine = body.get_text()

	newLine = re.sub('\.|,|<|>|\'|:|;|(|)|-|\*|/|\\|?|!|"|%|$|#', '', newLine)
	newLine = re.sub('Reuter', '', newLine)
	print "Punct: ", newLine
	return newLine
	
def process_removewords(newLine):
	newLine = newLine.lower()
	newLine = re.sub(' the | be | to | of | and | a | in | i | it | for | on | a | if | at | an | or | so | its ', ' ', newLine)

	newLine = re.sub('[0-9]+', 'NUMBER', newLine)
	print "Remove: ", newLine
	return newLine
	
def process_POS(newLine):
	newLine = newLine.split()
	newLine = nltk.pos_tag(newLine)
	outLine=""
	for tup in newLine:
		st = tup[0] + " " + tup[1]
		outLine = outLine + " " + st
	print "POS: ", outLine
	
	return outLine
	
def process_lemm(newLine):
	newLine = newLine.split()
	lmtzr = WordNetLemmatizer()
	iterator = len(newLine)
	iterator = iterator[::2]
	for x in iterator:
		print "Lemmaing: ", newLine
		print "Specifically: ", newLine[x]
		newLine[x] = lmtzr.lemmatize(newLine[x])
	
	
	newLine = " ".join(newLine)
	print newLine
	return newLine

main()