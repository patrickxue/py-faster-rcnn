from IKEA.items import IkeaItem
import ipdb
import datetime
import scrapy
import graphlab as gl
from scrapy.http import HtmlResponse
from scrapy.selector import HtmlXPathSelector

cata = gl.SFrame()
synset_sf = gl.SFrame()

class IkeaSpider(scrapy.Spider):
  name = "IKEA_catalogue_synset_spider"
  start_urls = ["http://www.ikea.com/us/en/catalog/allproducts/"]
  #start_urls = ["http://www.ikea.com/us/en/catalog/products/S39129827/"]
  base_url = "http://www.ikea.com"

  def parse(self, response):
    url = response.css(".productCategoryContainer .textContainer a")
    
    for sub_cata in url:
      yield scrapy.Request(self.base_url + sub_cata.xpath("@href").extract_first(), self.parse_page)

  def parse_page(self, response):
    #for href in response.css(".product .image img"):
    for href in response.css(".product .image"):
      prd_url = self.base_url + href.css("a").xpath("@href").extract_first()
      #ipdb.set_trace()
      yield scrapy.Request(prd_url, self.parse_synset)
      #pid = prd_url.split('/')[-2]
      #img_url = self.base_url + href.css("img").xpath("@src").extract_first()
 
  def parse_synset(self, response):
    # parse synset for different colors, covers
    #pid = response.url.split("/")[-2]
    #prd_url = self.base_url + response.css("#productImg").xpath("@src").extract_first()
    synset = response.css("#cartTxt01 .borderMar").xpath("@id").extract() 
    global synset_sf
    synset_sf = synset_sf.append(gl.SFrame({"synset": [synset]}))
    if synset_sf.__len__() % 3000 == 0:
      synset_sf.save("synset_{}.gl".format(synset_sf.__len__()))
    if synset_sf.__len__() > 6610 and synset_sf.__len__()%10 == 0:
      synset_sf.save("synset_{}.gl".format(synset_sf.__len__()))
    #bb=aa[aa.apply(lambda x: x["synset"] != [])]
    #yield IkeaItem(file_urls=prd_url, synset=synset, pid=pid)
    #syn_l = map(lambda x: x.split("cart")[1], synset)
