import graphlab as gl
from IKEA.items import IkeaItem
import ipdb
import datetime
import scrapy
 
data = gl.SFrame()
class IkeaSpider(scrapy.Spider):
  '''run: scrapy crawl query_cata_spider -o query_cata_spider.json
  crawl each sub_cata gallery (query img) with matching catalogue images (serving as ground_truth)
  '''
  name = "query_cata_spider"
  start_urls = ["http://www.ikea.com/us/en"]
  #start_urls = ["http://www.ikea.com/us/en/catalog/categories/departments/food/"]
  base_url = "http://www.ikea.com/"

  def parse(self, response):
    xpath_list =  response.css(".departmentLinkBlock .linkContainer a")
    for sub_cata in xpath_list:
      yield scrapy.Request(self.base_url + sub_cata.xpath("@href").extract_first(), self.parse_gallery)

  def parse_gallery(self, response):
    xpath_view_gallery = response.css(".gridRow .gridComponent a")
    # only some of teh sub_cata contains view gallery option
    # select those contains View Gallery option
    if xpath_view_gallery is not None:
      href = xpath_view_gallery.xpath("@href").extract_first()
      # href points to gallery address
      if href[:3] == '/us':
        yield scrapy.Request(self.base_url + href, self.parse_gallery_grid)
 
  def parse_gallery_grid(self, response):
    gallery_grid = response.css(".roomblock a")
    for gallery_pic in gallery_grid:
      yield scrapy.Request(self.base_url + gallery_pic.xpath("@href").extract_first(), self.parse_gallery_pic)
      
  def parse_gallery_pic(self, response):
    ipdb.set_trace()
    query = response.css(".roomComponent img")  
    query_url = query.xpath("@src").extract_first()
    if query_url[:4] != "http":  # using relative address
      query_url = self.base_url +  query_url
    #yield scrapy.Request(query_url, self.parse_img)
    # parse catalogue
    cata = response.css(".product .image img")
    cata_url = cata.xpath("@src").extract_first()
    if cata_url[:4] != "http":
      cata_url = self.base_url + cata_url 
    #yield scrapy.Request(cata_url, self.parse_img)
    data = data.append(gl.SFrame({"query": [query_url], "cata": [cata_url]}))
    data.save("../../img_url.gl")
    all_url = [].append(query_url).append(cata_url)
    yield scrapy.Request(all_url, self.parse_img)
    
  def parse_img(self, response):
    img_url = response.url
    for url in img_url:
      yield IkeaItem(file_urls=[url])
