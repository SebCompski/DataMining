#! /usr/bin/env python

import re
import string
import nltk

from bs4 import BeautifulSoup

def main():
	count = 0
	print "Starting..."
	#Read in the document
	soup = BeautifulSoup(open("pre_processed_records.sgm"))
	
	#Get the list of stories
	articles = soup.findAll('story')
	
	for art in articles:
		count = count + 1
	
	print count
	
main()