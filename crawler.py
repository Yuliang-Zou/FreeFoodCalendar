import re, sys, os
import httplib
import urllib2
from urlparse import urlparse
from BeautifulSoup import BeautifulSoup, SoupStrainer
from sets import Set

reload(sys)
sys.setdefaultencoding('UTF8')

domain = "events.umich.edu"
blacklist = ['.pdf','.cgi','.jpg','.png','.bmp','.jpeg','.asp','.pls','.ics']


def verify(url):
	if (domain+'/event') in url:
		if url.rsplit('/', 1)[-1] != 'ical':
			return True
		
	return False
	
	
def is_html(url):
	if os.path.splitext(urlparse(url).path)[1] not in blacklist:
		return True
	return False

def crawler(urls, max_urls):
	crawled = Set()
	queued = Set(urls)
	pairs = []
	while urls and len(crawled) < max_urls:
		page=urls.pop(0)
		if is_html(page):
			if page not in crawled:
				try:
					print(page)
					links=BeautifulSoup(urllib2.urlopen(page,timeout=5).read(), parseOnlyThese=SoupStrainer('a'))
					for link in links:
						url = domain + link['href']
						if verify(url) and url not in queued:
							# print(url)
							urls.append('http://' +url)
							# print(urls)
							queued.add('http://' +url)
					# print(page)
					crawled.add(page)
					# print(crawled)
				except:
					continue
	return crawled,pairs
	
	
if __name__ == "__main__":
	orig_stdout = sys.stdout
	url_file = sys.argv[1]
	max_urls = int(sys.argv[2])
	crawled_urls = []
	with open(url_file, 'r') as urls_file:
		urls=urls_file.readlines()
		crawled,pairs = crawler(urls,max_urls)
	f = file('URLs.txt', 'w')
	sys.stdout = f
	for x in crawled:
		print x
	f = file('pairs.txt', 'w')
	sys.stdout = f
	for x in pairs:
		print x[0],x[1]
	sys.stdout = orig_stdout
	