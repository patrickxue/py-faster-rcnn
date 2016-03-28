"""end to end IKEA demo using crawled img. Run from the directory py-faster-RCNN/IKEA"""
import graphlab as gl
import numpy as np
import matplotlib.pyplot as plt
import _init_paths
from fast_rcnn.config import cfg
import caffe, os, sys
import argparse
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
  reference_label = topk_rois["reference_label"]
  synset = gl.load_sframe("./synset_clean.gl")["synset"]
  syn_id = map(lambda x: [x_i.strip("cart") for x_i in x], synset)
  refer_syn = []   # enlarged matched reference set
  for id in reference_label:
    refer_syn.append(id)
    for syn in syn_id:
      if id in syn:
        refer_syn.extend(syn)
  #matches = gl.SFrame({"pid": pid_GT}).join(topk_rois, on={"pid": "reference_label"}, how="inner")
  matches = set(pid_GT).intersection(set(refer_syn))
  recall = len(matches)
  recall_rate = recall/len(pid_GT)
  return matches, recall, recall_rate

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
  neighbors, db_sf, cand_sf = match.demo(net, query, db, NMS_THRESH_GLOBAL=0.6, SCORE_THRESH=100)
  #neighbors, db_sf, cand_sf = load_neighbors_features()
  neighbors = neighbors.add_row_number()
  #neighbors.print_rows()
  cata_dic_l = data[qid]["cata"]
  GT_db = map(lambda x: x["c_img"], cata_dic_l)
  GT_pid = map(lambda x: x["pid"], cata_dic_l)
  GT_sf = gl.SFrame()
  GT_sf["image"] = GT_db
  GT_sf["pid"] = GT_pid
  alexnet = "~/py-faster-rcnn/tools/alexnet.gl"
  dfe = gl.load_model(alexnet)
  GT_feat = dfe.transform(GT_sf)
  nn = gl.nearest_neighbors.create(GT_feat,label="pid", features=['deep_features.image'],distance='cosine', verbose=False)
  #GT_nn = nn.query(cand_sf,radius=0.6,k=1)
  GT_nn = nn.query(cand_sf, k=1, verbose=False)
  topk = 3*len(cata_dic_l)  # get the same num of images as in GT
  topk_rois = get_topRoI_distance(neighbors, topk=topk)
  #topk_rois.print_rows(max_column_width=20)
  # topk_group: SFrame with query_label and nearest neighbor list
  #topk_group = topk_rois.groupby(["query_label"], {"nn_l": gl.aggregate.CONCAT("reference_label")})

  # show the original query image
  gl.SFrame([query])["X1"].show()  # the img is small

  # show all ground truth images
  cata_img_sa = gl.SArray(map(lambda x: x["c_img"], cata_dic_l))
  cata_img_sa.show()  # ground truth

  matches_db = topk_rois.join(db_sf, on={"reference_label": "pid"}, how="inner")
  matches_roi_l = []
  for roi in set(matches_db["query_label"]):
    # shows the ground truth closest to the crop
    GT_matched = GT_nn[GT_nn['query_label']==roi].sort("rank")[0]
    GT_matched_Id = GT_matched["reference_label"]
    GT_matched_dist = GT_matched["distance"]

    # Show ground truth
    print 'Ground truth image:'
    GT_sf[GT_sf['pid'] == GT_matched_Id]['image'].show()
    print 'GT distance %s, %s = %f' % (roi, GT_matched_Id, GT_matched_dist)

    # Show matched cata img
    matches_roi = matches_db[matches_db["query_label"] == roi]
    matches_img_sa = gl.SArray(map(lambda x: x["image.1"], matches_roi))
    roi_cata_sa = gl.SArray([matches_roi["image"][0]]).append(matches_img_sa)
    roi_cata_sa.show()
    for row in matches_roi.sort("rank"):
        print "DB distance %s, %s = %f" % (row["query_label"],
                                           row["reference_label"],
                                           row["distance"])
    matches_roi_l.append(matches_roi)

  #matches_db_sf_l = map(lambda x, y: gl.SFrame([x]).append(gl.SFrame([y])), matches_db["image"], matches_db["image.1"])  # matched RoI and cata pairs
  #fig, ax = plt.subplots(figsize=(12, 12))
  #ax.imshow(query.pixel_data, aspect='equal')
  #show_img_list(matches_db_sf_l)
  #gl.SFrame([query])["X1"].show()  # the img is small
  #matches, recall, recall_rate = join(topk_rois, qid, data)
  #ipdb.set_trace()
  #matches.print_rows()

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
  caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                            NETS[args.demo_net][1])
  #caffemodel = os.path.join(cfg.DATA_DIR, '../output/faster_rcnn_end2end/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel')
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
  #full_db = gl.load_sframe("./feature_AlexNet_ImageNet_cropped.gl")  # only contain features
  full_db = gl.load_sframe("./cata_cls_img/arm_chairs.gl")  # only contain features
  cls = set(data["cls"])
  cls_sf = gl.SFrame({"cls": cls}).add_row_number()
  cls_sf.print_rows()
  q_cls_id = input(">>> input query class id: ")
  q_cls = cls_sf["cls"][q_cls_id]
  data_cls = data[data["cls"]==q_cls]
  qid = input(">>> input query id: 0~236: ")
  #qid = input(">>> input query id: 0~{}: ".format(data_cls.__len__())
  #qid = 0
  #demo(net, qid, data, full_db)
  demo(net, qid, data_cls, full_db)
