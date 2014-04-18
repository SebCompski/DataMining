#! /usr/bin/env python

import re
import string

from bs4 import BeautifulSoup

id = 0
f = open('smaller_processed_records.sgm', 'a')

def main():
	print "Starting..."
	#Read in the document
	soup = BeautifulSoup(open("pre_processed_records.sgm"))
	process_file(soup)

def output_to_file(output):
	global f
	try:
		f.write(output)
	except UnicodeEncodeError:
		print "HAHA, YOU'RE NOT STOPPING ME THIS TIME ERROR, FUCK YOU"
	
def process_file(soup):
	output = ""
	
	#For all of the body tags, take their strings, do the pre-processing on them
	articles = soup.findAll('story')
	
	#For each article, run the process article method
	for art in articles:
		print art
		process_article(art)
		
def process_article(articles):	
	outputs=""
	topic = articles.get('topics')
	#print topic
	if topic:
		rTopic = topic.get_text()
		print rTopic
		if rTopic == "earn" or rTopic == "acquisitions" or rTopic == "money-fx" or rTopic == "grain" or rTopic == "crude" or rTopic == "trade" or rTopic == "intrest" or rTopic == "ship" or rTopic == "wheat" or rTopic == "corn":
			outputs = str(articles) + "\n"
			print "in"
			#outputs = re.sub('\:|  -- :| - :|-NONE-', ' ', outputs)
			output_to_file(output)

main()