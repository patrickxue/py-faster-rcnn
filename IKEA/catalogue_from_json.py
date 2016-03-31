import graphlab as gl
import _init_paths
import numpy as np
from PIL import Image
from utils import from_pil_image as PIL2gl
from cStringIO import StringIO
import json
import urllib
import ipdb
import cv2

def scale_img(cata_db_img):
 # scaled imgs and store the scaled_imgs along with scale para
 scales = [0.5, 0.75, 1.5, 2]
 aug_db = gl.SFrame()
 for scale in scales:
   ipdb.set_trace()
   scale_db = gl.SFrame()
   imgs = cata_db_img["image_cropped"]
   scaled_imgs = map(lambda x: PIL2gl.from_pil_image(Image.fromarray(cv2.resize(x.pixel_data, (0, 0), fx=scale, fy=scale))), imgs)
   scale_db["image"] = scaled_imgs
   scale_db["scale"] = scale * np.ones(scale_db.__len__())
   scale_db["pid"] = cata_db_img["pid"]
   aug_db = aug_db.append(scale_db)
 return aug_db

alexnet = "~/py-faster-rcnn/tools/alexnet.gl"
dfe = gl.load_model(alexnet)
cata_db_img = gl.load_sframe("./cata_db_image_cropped.gl")
aug_db = scale_img(cata_db_img)
aug_db_feat = dfe.transform(aug_db) 

# download the real img from url SFrame
#cata_db = gl.load_sframe("./cata_db.gl")
#cata_db = gl.load_sframe("./cata_db_img.gl")
#cata_db.rename({"img": "image"})
#cata_db = gl.load_sframe("./feature_PLACE_db.gl")
#cata_db = gl.load_sframe("./cata_db_image_cropped.gl")
#cata_db.remove_column("image")
#cata_db.rename({"img_cropped": "image"})
cata_db = gl.SFrame.read_json("./cata_cls.json")
cata_db_img = gl.SFrame() 
# Get img from url
if "url" not in cata_db.column_names():
  url = map(lambda x: x[0], cata_db["file_urls"])
  cata_db["url"]=url
for cata in cata_db:
  sub_cata_img = PIL2gl.from_pil_image(Image.open(StringIO(urllib.urlopen(cata["url"]).read())))
  cata_db_img = cata_db_img.append(gl.SFrame({"url": [cata["url"]], "pid": [cata["pid"]], "image": [sub_cata_img]}))

# Get features from dfe
ipdb.set_trace()
feature_db = dfe.transform(cata_db_img)
feature_db.save("./cata_cls_img/arm_chairs.gl")
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
