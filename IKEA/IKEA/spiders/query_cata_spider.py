import graphlab as gl
from IKEA.items import IkeaItem
import ipdb
import datetime
import scrapy
 
data_url = gl.SFrame()
data = gl.SFrame()

class IkeaSpider(scrapy.Spider):
  '''run: scrapy crawl query_cata_spider -o query_cata_spider.json
  crawl each sub_cata gallery (query img) with matching catalogue images (serving as ground_truth)
  '''
  name = "query_cata_spider"
  start_urls = ["http://www.ikea.com/us/en"]
  #start_urls = ["http://www.ikea.com/us/en/catalog/categories/departments/food/"]
  base_url = "http://www.ikea.com"

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
    # response url contains the class (the img is from bedroom/bathroom, etc) info
    cls = response.url.split('/')[9]
    query = response.css(".roomComponent img")
    query_url = query.xpath("@src").extract_first()
    query_item = IkeaItem(file_urls=query_url)
    if query_url[:4] != "http":  # using relative address
      query_url = self.base_url +  query_url
    # parse catalogue
    cata = response.css(".product .image img")
    cata_url = []
    #cata_list = []
    ipdb.set_trace()
    for sub_cata in cata:
      sub_cata_url = sub_cata.xpath("@src").extract_first()
      if sub_cata_url[:4] != "http":
        sub_cata_url = self.base_url + sub_cata_url 
      cata_url.append(sub_cata_url)
      #cata_list = cata_list.append(IkeaItem(file_urls=sub_cata_url))
    data_url = data_url.append(gl.SFrame({"cls": [cls], "query": [query_url], "cata": [cata_url]}))
    #data.append(gl.SFrame({"cls": [cls], "query": [query_item], "cata": cata_list}))
    #yield scrapy.Request(all_url, self.parse_img)
    
  def parse_img(self, response):
    img_url = response.url
    query = IkeaItem(file_urls=img_url[0])
    cata_list = []
    for url in img_url[1:]:
      cata_list = cata_list.append(IkeaItem(file_urls=[url]))
    data.append(gl.SFrame({"cls": ["foo"], "query": [query], "cata": cata_list}))

data_url.save("../../data_url.gl")
#data.save("../../data_item.gl")
