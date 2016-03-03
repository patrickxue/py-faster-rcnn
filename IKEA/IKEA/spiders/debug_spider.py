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
  name = "debug_spider"
  start_urls = ["http://www.ikea.com/us/en"]
  #start_urls = ["http://www.ikea.com/us/en/catalog/categories/departments/bathroom/tools/coba/roomset/20161_coba02a/"]
  #start_urls = ["http://www.ikea.com/us/en/catalog/categories/departments/bathroom/"]
  base_url = "http://www.ikea.com"

  def parse(self, response):
    xpath_list =  response.css(".departmentLinkBlock .linkContainer a")
    for sub_cata in xpath_list:
      yield scrapy.Request(self.base_url + sub_cata.xpath("@href").extract_first(), self.parse_gallery)

  def parse_gallery(self, response):
    # response.url = http://www.ikea.com/us/en/catalog/categories/departments/bathroom/
    xpath_view_gallery = response.css(".gridRow .gridComponent .bodyTextGray")
    # only some of teh sub_cata contains view gallery option
    # select those contains View Gallery option
    if xpath_view_gallery is not None:
      href = xpath_view_gallery.xpath("//a[contains(., 'View gallery')]/@href").extract_first()
      # href points to gallery address
      if href[-4:] == 'set/':  # ensure it points to roomset page
        yield scrapy.Request(self.base_url + href, self.parse_gallery_grid)
 
  def parse_gallery_grid(self, response):
    # response.url= http://www.ikea.com/us/en/catalog/categories/departments/bathroom/tools/coba/roomset/
    gallery_grid = response.css(".roomblock a")
    for gallery_pic in gallery_grid:
      href = gallery_pic.xpath("@href").extract_first()
      yield scrapy.Request(self.base_url + href, self.parse_gallery_pic)
      
  def parse_gallery_pic(self, response):
    # response.url=http://www.ikea.com/us/en/catalog/categories/departments/bathroom/tools/coba/roomset/20161_coba02a/
    # response url contains the class (the img is from bedroom/bathroom, etc) info
    cls = response.url.split('/')[8]
    query = response.css(".roomComponent img")
    query_url = query.xpath("@src").extract_first()
    if query_url is not None and query_url[:4] != "http":  # using relative address
      query_url = self.base_url +  query_url
    # parse catalogue
    cata = response.css(".product .image img")
    prd = response.css(".product .image a")
    cata_url = []
    for idx in xrange(len(cata)):
      sub_cata_url = cata[idx].xpath("@src").extract_first()
      if sub_cata_url[:4] != "http":
        sub_cata_url = self.base_url + sub_cata_url
      sub_prd_url = prd[idx].xpath("@href").extract_first()
      sub_prd_id = sub_prd_url.split('/')[-2]
      sub_cata_dic = {"url": sub_cata_url, "prd_id": sub_prd_id}
      cata_url.append(sub_cata_dic)

    global data_url  # make it to refer to the global SFrame variable for continuous appending
    data_url = data_url.append(gl.SFrame({"cls": [cls], "query": [query_url], "cata": [cata_url]}))
    if data_url.__len__()%100 == 0:
      data_url.save("../../data_url_snapshot_{}.gl".format(data_url.__len__()))
    if data_url.__len__() > 235:
      data_url.save("../../data_url_snapshot_{}.gl".format(data_url.__len__()))
    
  def closed():   
    data_url.save("../../data_url_final.gl")
#data_url.save("../../data_url_final.gl")
