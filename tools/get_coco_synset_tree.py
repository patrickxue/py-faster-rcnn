import os
import urllib2
import numpy as np
import graphlab as gl
import ipdb
import cPickle as pkl

# get wnids corresponding to coco categories
#coco_sf = gl.SFrame()
#coco_wnids = [l.strip() for l in open("./coco_synset.txt").readlines()] 
#for ent in coco_wnids:
#  sp = ent.split(" ")
#  cid = sp[0]
#  wnid = sp[-1]
#  cls = ent.strip(cid).strip(wnid).strip()
#  coco_sf = coco_sf.append(gl.SFrame({"id": [cid], "wnid": [wnid], "cls": [cls]}))

coco_sf = gl.load_sframe("./coco_wnid.gl")
wnid_txt = pkl.load(open("./wnid_txt_synset.pkl"))
wnids_1k = wnid_txt.keys() 

coco_imnet = gl.SFrame()
for ent in coco_sf:
  wnid = ent["wnid"]
  children = []
  if wnid != 'None':
    req = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=' + wnid + '&full=1'
    res = urllib2.urlopen(req)
    children = [child.strip('-').strip('\r\n') for child in res.readlines()]
    idx = map(lambda x: x in wnids_1k, children)
    idx_1k = np.where(idx)[0]
    children_1k = []
    if len(idx_1k) > 0:
      children_1k = np.array(children)[idx_1k]
      coco_imnet = coco_imnet.append(gl.SFrame({"cls": [ent["cls"]], "in_im": [True], "child": [children_1k]}))
    else:
      coco_imnet = coco_imnet.append(gl.SFrame({"cls": [ent["cls"]], "in_im": [False], "child": [children_1k]}))
  else:
    coco_imnet = coco_imnet.append(gl.SFrame({"cls": [ent["cls"]], "in_im": [False], "child": [children]}))

coco_imnet["num_ch"]=map(lambda x: len(x), coco_imnet["child"])
ipdb.set_trace()

#imagenet_map = {'wnid': 'cls'}
#
#coco_tree = gl.SFrame()
#for cls in CLASSES:
#  coco_cls_children = im.get_children(cls)
#  cls_imnet = []
#  if len(coco_cls_children) == 0:
#    sf = gl.SFrame({'in_imagenet': False, 'coco_cls': cls, 'imagenet_cls': [cls_imnet]})
#  else:
#    for wnid in coco_cls_children:
#      in_imagenet = imagenet_map.get(wnid, None)
#      if in_imagenet is not None:
#        cls_imnet.append(wnid + in_imagenet)
#    if len(cls_imnet) == 0:
#      sf = gl.SFrame({'in_imagenet': False, 'coco_cls': cls, 'imagenet_cls': [cls_imnet]})
#    else:
#      sf = gl.SFrame({'in_imagenet': True, 'coco_cls': cls, 'imagenet_cls': [cls_imnet]})
#
###############################string  matching subclassing
#def in_imagenet(cls, imagenet_synset):
#  idx  = map(lambda x: cls.lower() in x.lower(), imagenet_synset)
#  return np.array(idx)
#
#ipdb.set_trace()
#for cls in CLASSES:
#  cls_imnet = []
#  idx = in_imagenet(cls, imagenet_synset)
#  if np.sum(idx) > 0:
#    cls_imnet = imagenet_synset[idx]
#    sf = gl.SFrame({'in_imagenet': True, 'coco_cls': cls, 'imagenet_cls': [cls_imnet]})
#  else:
#    sf = gl.SFrame({'in_imagenet': False, 'coco_cls': cls, 'imagenet_cls': [cls_imnet]})
#
#  coco_tree = coco_tree.append(sf)
#
#coco_tree.save('./coco_synset_tree.gl')
