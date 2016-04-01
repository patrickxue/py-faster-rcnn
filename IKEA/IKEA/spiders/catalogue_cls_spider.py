from IKEA.items import IkeaItem
import ipdb
import datetime
import scrapy
import graphlab as gl

cata = gl.SFrame()

class IkeaSpider(scrapy.Spider):
  name = "IKEA_catalogue_cls_spider"
  start_urls = ["http://www.ikea.com/us/en/catalog/allproducts/"]
  start_urls = ["http://www.ikea.com/us/en/catalog/categories/departments/living_room/16239/"]
  cls = "arm_chairs"
  base_url = "http://www.ikea.com"

  def parse_default(self, response):
    url = response.css(".productCategoryContainer .textContainer a")
    #cls = response.css(".productCategoryContainer .textContainer a//text()")
    #cls_url = zip(csl, url)
    for sub_cata in url:
      yield scrapy.Request(self.base_url + sub_cata.xpath("@href").extract_first(), self.parse_page)

  def parse(self, response):
    #http://www.ikea.com/us/en/catalog/categories/departments/living_room/16239/
    #for href in response.css(".product .image img"):
    for href in response.css(".product .image"):
      prd_url = self.base_url + href.css("a").xpath("@href").extract_first()
      pid = prd_url.split('/')[-2]
      img_url = self.base_url + href.css("img").xpath("@src").extract_first()
      yield IkeaItem(file_urls=[img_url], pid=[pid])
