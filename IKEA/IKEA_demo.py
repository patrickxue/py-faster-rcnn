"""end to end IKEA demo using crawled img. Run from the directory py-faster-RCNN/IKEA"""
import graphlab as gl
import _init_paths
from fast_rcnn.config import cfg
import caffe, os, sys
import argparse
import ipdb
import rpn_matching as match

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

def get_topk_match(neighbors, topk=5):
  # get topk rois according to roi score
  neighbors["roi_score"]
  return topk_rois

def join(topk_rois, data):
  cata_GT = data[qid]["cata"]
  topk_cand = []
  for roi in topk_rois:
    topk_cand.append(roi["cata"])
  matches = inner_join(topk_cand, cata_GT)
  return matches

def demo(net, query, db):
  neighbors, db_sf, cand_sf = match.demo(net, query, db)
  neighbors = neighbors.add_row_number()
  neighbors.print_rows()
  roi_id = input(">>> input roi_id: ")
  match.image_join(roi_id, db_sf, cand_sf)
  topk_rois = get_topk_match(neighbors)

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
  full_db = gl.load_sframe("./feature_PLACE_db.gl")
  #dfe = gl.load_model("./PLACE.gl")
  cls = list(set(data["cls"]))
  #qid = input(">>> input query id (0~237): ")
  query = data[0]["q_img"]
  demo(net, query, small_db)
