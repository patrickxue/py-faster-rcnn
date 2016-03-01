from IKEA.items import IkeaItem
import ipdb
import datetime
import scrapy
 
class IkeaSpider(scrapy.Spider):
  name = "IKEA_catalogue_spider"
  start_urls = ["http://www.ikea.com/us/en/catalog/allproducts/"]
  base_url = "http://www.ikea.com"

  def parse(self, response):
    url = response.css(".productCategoryContainer .textContainer a")
    
    for sub_cata in url:
      yield scrapy.Request(self.base_url + sub_cata.xpath("@href").extract_first(), self.parse_page)

  def parse_page(self, response):
    for href in response.css(".product .image img"):
      yield scrapy.Request(self.base_url + href.xpath("@src").extract_first(),
        self.parse_img)
 
  def parse_img(self, response):
    img_url = response.url
    yield IkeaItem(file_urls=[img_url])
