import scrapy
from datetime import datetime
import pandas as pd
import os
from shutil import rmtree

class EuDataSpiderSpider(scrapy.Spider):
    name = "eu_data_spider"        
    def start_requests(self):
        '''
            Select the URLs to extract from
        '''      
        # Get the output dir for the data from the args
        self.output_directory = getattr(self, 'output_directory', 'output/')  
        # Delete the output dir if exists and create anew
        try:
            if os.path.isdir(self.output_directory):
                rmtree(self.output_directory)
            os.makedirs(self.output_directory)  
        except Exception as e:
            print(e)
        # Process the urls and other data from the file
        path_to_file = getattr(self, 'path_to_file', 'data.csv')

        for index, row in pd.read_csv(path_to_file).iterrows():
            if isinstance(row.html_to_download, str):
                yield scrapy.Request(
                                        url = row.html_to_download, 
                                        callback = self.parse, 
                                        meta = {'id':row.id, 'title':row.title_, 'authors':row.authors, 'date':row.date_document, 'celex':row.celex,'Full_OJ':row. Full_OJ}
                                    )

    def parse(self, response):
        print(f"Processing.. {response.url}")
        
        # Find the relevant paragraphs in the html documents
        paragraphs = response.xpath('//p/text()|//span/text()|//h1/text()|//h2/text()|//h3/text()').extract()
        with open(f"{self.output_directory}{response.meta['id']}", 'w') as f:
            for paragraph in paragraphs:
                try:
                    f.write(paragraph)
                except Exception as e:
                    # If a paragraph encounters an error, ignore it
                    pass
                f.write('\n')
        
        yield dict(
                    id = response.meta['id'],
                    title = response.meta['title'],
                    url = response.url,
                    authors = response.meta['authors'], 
                    date = response.meta['date'], 
                    celex = response.meta['celex'], 
                    Full_OJ = response.meta['Full_OJ'],
                    timeStamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                )
