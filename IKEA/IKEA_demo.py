"""end to end IKEA demo using crawled img. Run from the directory py-faster-RCNN/IKEA"""
import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
import _init_paths
from fast_rcnn.config import cfg
import caffe, os, sys
import argparse
import shutil
import ipdb
import rpn_matching as match

gl.canvas.set_target('ipynb')
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# only IKEA related classes
#CLASSES = ('__background__', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def get_topRoI_score(neighbors, cand_sf_withScore, topk=5):
  # get topk rois according to roi score
  score = cand_sf_withScore["score"]
  id_score = get_id(score)  # TODO
  top_rois = neighbors[neighbors.apply(lambda x: True if  x["query_label"] in id_score else False)] 
  return topk_rois

def get_topRoI_distance(neighbors, topk=5):
  """get topk RoIs according to distance"""
  dist = np.asarray(neighbors["distance"])
  #query_label = np.asarray(neighbors["query_label"])
  idx = dist.argsort()
  idx_sf = gl.SFrame({"id": idx})[0:topk]
  topk_rois = neighbors.join(idx_sf, on="id", how="inner")
  return topk_rois

def join(topk_rois, qid, data):
  cata_GT = data[qid]["cata"]
  pid_GT = map(lambda x: x["pid"], cata_GT)
  matches = gl.SFrame({"pid": pid_GT}).join(topk_rois, on={"pid": "reference_label"}, how="inner")
  return matches

def load_neighbors_features():
  """load precomputed data, candidate RoIs is generated for query image 0"""
  neighbors_score = gl.load_sframe("./neighbors_features/neighbors_score.gl") 
  db_sf = gl.load_sframe("./neighbors_features/db_sf.gl")
  cand_sf = gl.load_sframe("./neighbors_features/cand_sf_qid=0.gl")
  return neighbors_score, db_sf, cand_sf

def show_img_list(img_l, col_name="X1"):
  for img in img_l:
    img[col_name].show()

def demo(net, qid, data, db):
  query = data[qid]["q_img"]
  #query = db[100]["image"]
  neighbors, db_sf, cand_sf = match.demo(net, query, db, SCORE_THRESH=100)
  #neighbors, db_sf, cand_sf = load_neighbors_features() 
  neighbors = neighbors.add_row_number()
  neighbors.print_rows()
  cata_dic_l = data[qid]["cata"]
  topk = 3*len(cata_dic_l)  # get the same num of images as in GT
  topk_rois = get_topRoI_distance(neighbors, topk=topk)
  #topk_rois.print_rows(max_column_width=20)
  crop_img = gl.image_analysis.load_images("./crop_buff/")
  shutil.rmtree("./crop_buff")
  id = map(lambda x: int(x["path"].split("/")[-1].split(".")[0].split("_")[1]), crop_img)
  crop_img["id"] = id
  #crop_img.remove_column("path")
  topk_rois = topk_rois.join(crop_img, on={"query_label": "id"}, how="inner") # append the image bytes
  # topk_group: SFrame with query_label and nearest neighbor list
  #topk_group = topk_rois.groupby(["query_label"], {"nn_l": gl.aggregate.CONCAT("reference_label")})
  db_img = gl.load_sframe("./cata_db_image.gl")
  matches_db = topk_rois.join(db_img, on={"reference_label": "pid"}, how="inner")
  matches_roi_l = []
  for roi in set(matches_db["query_label"]):
    matches_roi = matches_db[matches_db["query_label"] == roi]
    matches_img_sa = gl.SArray(map(lambda x: x["image.1"], matches_roi))
    roi_cata_sa = gl.SArray([matches_roi["image"][0]]).append(matches_img_sa)
    roi_cata_sa.show()  # show crop and matched cata img
    matches_roi_l.append(matches_roi)
  #matches_db_sf_l = map(lambda x, y: gl.SFrame([x]).append(gl.SFrame([y])), matches_db["image"], matches_db["image.1"])  # matched RoI and cata pairs
  #fig, ax = plt.subplots(figsize=(12, 12))
  #ax.imshow(query.pixel_data, aspect='equal')
  cata_img_sa = gl.SArray(map(lambda x: x["c_img"], cata_dic_l))
  #show_img_list(matches_db_sf_l)
  gl.SFrame([query])["X1"].show()  # the img is small
  cata_img_sa.show()  # ground truth
  ipdb.set_trace()
  # inner join with GT
  matches = join(topk_rois, qid, data)
  matches.print_rows()

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                      default=0, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
                      help='Use CPU mode (overrides --gpu)',
                      action='store_true')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                      choices=NETS.keys(), default='vgg16')

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals

  args = parse_args()

  prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                          'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
  #caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
  #                          NETS[args.demo_net][1])
  caffemodel = os.path.join(cfg.DATA_DIR, '../output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel')
  #caffemodel = os.path.join(cfg.DATA_DIR, '../output/LSDA_200_strong_detector_finetune_ilsvrc13_val1+train1k_iter_50000.caffemodel')
  #cfg.TEST.HAS_RPN = False # Use RPN for proposals

  if not os.path.isfile(caffemodel):
      raise IOError(('{:s} not found.\nDid you run ./data/script/'
                     'fetch_faster_rcnn_models.sh?').format(caffemodel))

  if args.cpu_mode:
      caffe.set_mode_cpu()
  else:
      caffe.set_mode_gpu()
      caffe.set_device(args.gpu_id)
      cfg.GPU_ID = args.gpu_id
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)

  print '\n\nLoaded network {:s}'.format(caffemodel)
  # para inint
  topk = 5
  # +++++load data, select query: cls, qid
  data = gl.load_sframe("./data_237.gl")
  #small_db = gl.load_sframe("../tools/features_sframe.gl")
  #full_db = gl.load_sframe("./feature_PLACE_db.gl")  # only contain features
  #full_db = gl.load_sframe("./feature_AlexNet_ImageNet_db.gl")  # only contain features
  full_db = gl.load_sframe("./features_GoogLeNet_PLACES_db.gl")  # only contain features
  #dfe = gl.load_model("./PLACE.gl")
  #cls = list(set(data["cls"]))
  qid = input(">>> input query id (0~236): ")
  demo(net, qid, data, full_db)
