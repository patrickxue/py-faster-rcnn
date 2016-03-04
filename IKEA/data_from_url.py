import graphlab as gl
import _init_paths
from PIL import Image
from utils import from_pil_image as PIL2gl
from cStringIO import StringIO
import urllib
import ipdb
data_url = gl.load_sframe("../../data_url_snapshot_237.gl")
data = gl.load_sframe("./data_237.gl")
cls_set = set(data_url["cls"])
ipdb.set_trace()
#print len(sum(data_url["cata"],[]))
data = gl.SFrame()
for cls in cls_set:
  print "+++++++++++++++++++++++++++++++++++++++++++processing: " + cls
  #cls = "living_room"
  data_url_cls = data_url[data_url["cls"] == cls]
  data_url_cls = data_url_cls.add_row_number("qid")
  for qry in data_url_cls:
    #q_img = urllib.urlretrieve(qry["query"], str(qry["qid"])+".jpg")
    q_url = qry["query"] 
    print "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq query_url: " + q_url
    if "miss" in q_url.split('/')[-1]:
      continue
    q_img = PIL2gl.from_pil_image(Image.open(StringIO(urllib.urlopen(q_url).read())))
    cata_list = []
    for sub_cata in qry["cata"]:
      c_url = sub_cata["url"]
      #print "ccccccccccccccccccccccccccccccccccccccc c_url: " + c_url
      if "miss" in c_url.split('/')[-1]:
        continue
      sub_cata_img = PIL2gl.from_pil_image(Image.open(StringIO(urllib.urlopen(c_url).read())))
      cata_list.append({"pid": sub_cata["prd_id"], "c_url": c_url, "c_img": sub_cata_img})

    data = data.append(gl.SFrame({"cls": [cls], "q_url": [q_url], "q_img": [q_img], "cata": [cata_list]}))

  data.save("./data_{}.gl".format(data.__len__()))
