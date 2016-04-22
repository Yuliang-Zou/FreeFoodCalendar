import os
import time
import threading
import sys


time_interval = float(sys.argv[1])
def crawl_and_scape():
	while True:
	#print datetime.datetime.now()
		os.system('python crawler.py url_starts.txt 200')
		os.system('scrapy crawl eventsfood')
		time.sleep(time_interval)

crawl_and_scape()



