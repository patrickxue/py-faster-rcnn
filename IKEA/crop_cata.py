import graphlab as gl
import _init_paths
import numpy as np
import ipdb
from utils import from_pil_image as PIL2gl
from PIL import Image

def crop_bg(img):
  # crop off irrelevant white background
  img_array = img.pixel_data[:, :, 0]
  idx = np.where(255 - img_array > 0)
  y = min(idx[0])
  y1 = max(idx[0])
  x = min(idx[1])
  x1 = max(idx[1])
  img_cropped = img.pixel_data[y:y1, x:x1, :]
  crop_gl_img = PIL2gl.from_pil_image(Image.fromarray(img_cropped))
  return crop_gl_img

cata_db = gl.load_sframe("./cata_db_image.gl")
imgs = cata_db["image"]
#img_cropped = map(crop_bg, imgs)
cnt = 0
img_cropped = []
failed = []
for img in imgs:
  try:
    img_cropped.append(crop_bg(img))
  except: 
    print "failed processing img: \n" + str(cnt)
    ipdb.set_trace()
    crop_bg(img)
    failed.append(img)
  cnt += 1
ipdb.set_trace()
cata_db["img_cropped"] = img_cropped 
cata_db.save("./cata_db_image_cropped.gl")
