# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class involveEvent(scrapy.Item):
	# define the fields for your item here like:
	description = scrapy.Field()
	url = scrapy.Field()
	pass

	
class eventsEvent(scrapy.Item):
	description = scrapy.Field()
	month = scrapy.Field()
	day = scrapy.Field()
	url = scrapy.Field()
	title = scrapy.Field()
	pass
