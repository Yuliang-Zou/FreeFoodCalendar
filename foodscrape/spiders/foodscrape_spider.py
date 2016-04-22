
import scrapy
from .. import settings
from scrapy.spiders import Spider
from scrapy.selector import HtmlXPathSelector
from foodscrape.items import involveEvent, eventsEvent
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup, SoupStrainer
import re, sys, os
import httplib
import urllib2
from classify import classify
from urlparse import urlparse

date_dict = {'Apr': 'April',
			'Mar':'March',
			'Jun':'June',
			'Jul':'July',
			'Aug':'August',
			'Sep':'September',
			'Oct':'October',
			'Nov':'November',
			'Dec':'December',
			'Jan':'January',
			'Feb':'February'}
			

regex_date = r"([a-zA-Z]+ \d+)"
regex_num = r"\d+"
domain = 'events.umich.edu/event/'

class InvolvementSpider(Spider):
	name = "involvementfood"
	allowed_domains = ["campusinvolvement.umich.edu"]
	start_urls = ["https://campusinvolvement.umich.edu/late-night-programs"
	"https://campusinvolvement.umich.edu/late-night-programs?page=1",
	"https://campusinvolvement.umich.edu/late-night-programs?page=2",
	"https://campusinvolvement.umich.edu/late-night-programs?page=3",
	"https://campusinvolvement.umich.edu/late-night-programs?page=4"]
	def parse(self, response):
		items =  response.xpath('//p/text()')
		for item in items:
			my_item = involveEvent()
			my_item['description'] =  item.extract()
			my_item['url'] = response.url
			yield my_item
			
			
class EventsSpider(Spider):
	name = "eventsfood"
	allowed_domains = ["events.umich.edu"]	
	start_urls = []
	with open('URLs.txt', 'r') as urls_file:
		start_urls=urls_file.readlines()
	
	def parse(self, response):
		my_item = eventsEvent()
		# my_item['title'] = response.xpath('//title/text()').extract()
		description =  response.xpath('//div[@class="event-description"]/text()').extract()
		# join(desc, opt)
		## desc: event description
		## opt = -1: no classifier
		## opt = 0: whitelist
		## opt = 1: naive bayes
		## opt = 2: svm
		## opt = 3: knn
		## opt = 4: decision tree
		if classify(" ".join(description),3):
			date_extract =  response.xpath('//time/text()').extract()
			month = re.findall(regex_date, date_extract[0], flags=0)
			my_item['month'] = date_dict[month[0][:3]]
			my_item['day'] = int(re.findall(regex_num, date_extract[0])[0])
			my_item['url'] = response.url
			yield my_item
			
		