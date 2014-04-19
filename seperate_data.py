#! /usr/bin/env python

import re
import string
import nltk

from bs4 import BeautifulSoup

train = open('train_data.sgm', 'a')
test = open('test_data.sgm', 'a')

def main():
	print "Starting..."
	#Read in the document
	soup = BeautifulSoup(open("pre_processed_records.sgm"))
	
	articles = soup.find_all('story')
	
	for art in articles:
		sort_articles(art)
		
def sort_articles(body):
	type = body.find("lewissplit")
	type = type.get_text()
	#print type
	if type=="TRAIN":
		write_train(str(body))
	elif type=="TEST":
		write_test(str(body))

def write_train(output):
	global train
	try:
		train.write(output)
	except UnicodeEncodeError:
		print "HAHA, YOU'RE NOT STOPPING ME THIS TIME ERROR, FUCK YOU"

def write_test(output):
	global test
	try:
		test.write(output)
	except UnicodeEncodeError:
		print "HAHA, YOU'RE NOT STOPPING ME THIS TIME ERROR, FUCK YOU"
		
main()