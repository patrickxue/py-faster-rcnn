import graphlab as gl
import _init_paths
from PIL import Image
from utils import from_pil_image as PIL2gl
from cStringIO import StringIO
import json
import urllib
import ipdb

# download the real img from url SFrame
#cata_db = gl.load_sframe("./cata_db.gl")
#cata_db = gl.load_sframe("./cata_db_img.gl")
#cata_db.rename({"img": "image"})
#cata_db = gl.load_sframe("./feature_PLACE_db.gl")
cata_db = gl.load_sframe("./cata_db_image_cropped.gl")
cata_db.remove_column("image")
cata_db.rename({"img_cropped": "image"})
alexnet = "~/py-faster-rcnn/tools/alexnet.gl"
dfe = gl.load_model(alexnet)
ipdb.set_trace()
feature_db = dfe.transform(cata_db)
cata_db_img = gl.SFrame() 
for cata in cata_db:
  sub_cata_img = PIL2gl.from_pil_image(Image.open(StringIO(urllib.urlopen(cata["url"]).read())))
  cata_db_img = cata_db_img.append(gl.SFrame({"url": [cata["url"]], "pid": [cata["pid"]], "img": [sub_cata_img]}))

cata_db_img.save("cata_db_img.gl")

ipdb.set_trace()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++   
# from json to SF and remove duplicatoin
cata_db = gl.SFrame.read_json("./cata_db_pid.json")
#pid_list = set(cata_db.apply(lambda x: x["pid"][0]))
#cata_db_dedup = cata_db[cata_db.apply(lambda x: True if x["pid"][0] in pid_list else False)]
cata_db_dedup = gl.SFrame()
pid_list = cata_db[0]["pid"]
cata_db_dedup = cata_db_dedup.append(gl.SFrame({"url": cata_db[0]["file_urls"], "pid": cata_db[0]["pid"]}))
for cata in cata_db:
  pid = cata["pid"][0]
  if pid in pid_list:
    continue
  pid_list.append(pid)
  print cata["pid"][0] + "\n++++++++++++++++++++++++"
  cata_db_dedup = cata_db_dedup.append(gl.SFrame({"url": cata["file_urls"], "pid": cata["pid"]}))
cata_db = json.loads("./cata_db_pid.json")
db_sf = gl.SFrame.read_json("./cata_db_pid.json")

cata_db_dedup.save("cata_db.gl")
#cata_db_new = gl.data_matching.deduplication.create(cata_db)
