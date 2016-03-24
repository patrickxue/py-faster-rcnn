import graphlab as gl
import ipdb

cata_db = gl.load_sframe("./cata_db_image.gl")
ipdb.set_trace()
imgs = cata_db["image"]
img_cropped = map(crop_bg, imgs)
cata_db["img_cropped"] = img_cropped 
cata_db.save("./cata_db_image_cropped.gl")

def crop_bg(img):
  # crop off irrelevant white background
  img_cropped = img.pixel_data
  return img_cropped
