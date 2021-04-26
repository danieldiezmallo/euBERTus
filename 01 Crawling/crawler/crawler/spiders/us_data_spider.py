import scrapy
from datetime import datetime
import os
from shutil import rmtree
import requests
import textract
import json

class UsDataSpiderSpider(scrapy.Spider):
    name = "us_data_spider"        
    def start_requests(self):
        '''
            Select the URLs to extract from
        '''      
        start_url = 'https://www.govinfo.gov/features/coronavirus'
        # Get the output dir for the data from the args
        self.output_directory = getattr(self, 'output_directory', 'output/')  
        try:
            if os.path.isdir(self.output_directory):
                rmtree(self.output_directory)
            os.makedirs(self.output_directory)  
        except Exception as e:
            print(e)

        self.tmp_directory = getattr(self, 'tmp_directory', 'scrapy_tmp/')
        
        yield scrapy.Request(url = start_url, callback = self.parse)  

    def parse(self, response):
        def file_text_parser(url):
            '''
                Fetch the pdf file from the url, save  it, process it with textract and return the text from the file and its name
            '''
            if os.path.isdir(self.tmp_directory):
                rmtree(self.tmp_directory)
            os.makedirs(self.tmp_directory)
            path = f"{self.tmp_directory}/tmp.pdf"
            try:
                open(path, 'wb').write(requests.get(url, allow_redirects=True).content)
            except Exception as e:
                print(e)
            try:
                text = textract.process(path, encoding='ascii').decode('ascii') 
            except Exception as e:
                print(e)
                text = ''
            rmtree(self.tmp_directory)
            title = url.split('/')[-1]
            return text, title

        print(f"Processing.. {response.url}")
        
        # Find the links in the page and recreate them from the original domain
        urls = response.xpath('//a[@class="underlined-link"]/@href').extract()
        urls = [f'https://www.govinfo.gov/{url}' for url in urls]
        
        # Get the text from the PDFs
        for url in urls:
            if requests.get(url).headers['Content-Type'].lower().split('/')[-1]=='pdf':
                text, title = file_text_parser(url)
                idx = len(os.listdir(self.output_directory))+1
                # Save the text
                with open(f"{self.output_directory}{idx}", 'w') as f:
                    for paragraph in text.split('\n'):
                        try:
                            f.write(paragraph)
                        except Exception as e:
                            # If a paragraph encounters an error, ignore it
                            pass
                        f.write('\n')  
                
                # As the results are not yielded every time, save them manually in the metadata file
                result = dict(
                            id = idx,
                            title = title,
                            timeStamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S"),
                        )  
                with open(f"{'/'.join(self.output_directory.split('/')[:-2])}/metadata.json", 'a') as f:
                    f.write(json.dumps(result))
                    f.write('\n')   

                yield result
