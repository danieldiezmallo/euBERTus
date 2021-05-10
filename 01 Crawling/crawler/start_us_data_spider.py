from scrapy.crawler import CrawlerProcess
from crawler.spiders.us_data_spider import UsDataSpiderSpider

process = CrawlerProcess({
'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
'FEED_FORMAT': 'json',
'FEED_URI': '../../data/01_crawled/us_data/metadata.json'
})

process.crawl(UsDataSpiderSpider, output_directory='../../data/01_crawled/us_data/text/')
process.start()