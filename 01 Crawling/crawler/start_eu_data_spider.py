from scrapy.crawler import CrawlerProcess
from crawler.spiders.eu_data_spider import EuDataSpiderSpider

process = CrawlerProcess({
'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)',
'FEED_FORMAT': 'json',
'FEED_URI': '..\\..\\data\\01_crawled\\eu_data\\metadata.json'
})

process.crawl(EuDataSpiderSpider, 
                                    output_directory='..\\..\\data\\01_crawled\\eu_data\\text\\',
                                    path_to_file='..\\..\\data\\00_raw\\extracted\\eu_data_preprocessed.csv')

process.start()