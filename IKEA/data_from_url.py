import graphlab as gl
import numpy as np
import urllib3
import ipdb
data_url = gl.load_sframe("../../data_url_snapshot_237.gl")
cls_set = set(data_url["cls"])
#print len(sum(data_url["cata"],[]))
data = gl.SFrame()
ipdb.set_trace()
for qry in data_url:
  q_img = urllib3.request(qry["query"])
  cls = qry["cls"]
  cata_list = []
  for sub_cata in qry["cata"]:
    sub_cata_img = urllib3.request(subcata["cata"])
    cata_list.append({"prd_id": sub_cata["prd_id"], "cata": sub_cata_img})
    data = data.append(gl.SFrame({"cls": [cls], "qeury": [q_img], "cata": [cata_list]}))

data.save("./data_{}.gl".format(data.__len__))
