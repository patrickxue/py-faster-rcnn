#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import ipdb
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import graphlab as gl
from PIL import Image
from utils import from_pil_image as PIL2gl
gl.canvas.set_target('ipynb')

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

## only IKEA related classes
#CLASSES = ('__background__', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor')
NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
def get_cdf(dets_nms_all, CONF_THRESH=np.linspace(0,1,11)):
    cdf = gl.SFrame()
    for conf in CONF_THRESH:
        num_rois = np.sum(dets_nms_all[:, 4] >= conf)
        cdf = cdf.append(gl.SFrame({"conf": [conf], "num_rois": [num_rois]})) 
    return cdf
def transform_and_build_nn(cand_sf, dfe, db="./features_sframe.gl", radius=0.51, k=1):   
    cand_sf = dfe.transform(cand_sf)
    cand_sf = cand_sf.add_row_number()
    db_sf = gl.SFrame(db)
    db_sf = db_sf.add_row_number()
    nn = gl.nearest_neighbors.create(db_sf,label="pid", features=['deep_features.image'],distance='cosine')
    neighbors = nn.query(cand_sf,radius=radius,k=k)
    ipdb.set_trace()
    neighbors_score = neighbors.join(cand_sf, on="id", how="inner")
    #neighbors_img = neighbors_score.join(cand_sf, on={"reference_label": "pid"}, how="inner")
    return neighbors_score, db_sf, cand_sf

def image_join(neighbors, db_sf, cand_sf, query_id):
    """"append the matched catalogue img with a query RoI"""
    tmp_nn = neighbors[neighbors['query_label'] == query_id]
    tmp_db = db_sf.filter_by(tmp_nn['reference_label'], 'id')
    cand = cand_sf[cand_sf['id'] == query_id]
    return cand.append(tmp_db)

def save_img_SF(img, rois):
    """save imgs as SFrame"""
    cand_sf = gl.SFrame()
    for roi in rois:    
        #cropped = img[y:y+h, x:x+w, :]
        cropped = img[roi[1]:roi[3], roi[0]:roi[2], :]
        cropped_img = PIL2gl.from_pil_image(Image.fromarray(cropped))
        #scipy.misc.imsave('crop_%d.jpg'%count, cropped)
        cur_sf = gl.SFrame({'image': [cropped_img], 'score': [roi[4]]})
        cand_sf = cand_sf.append(cur_sf)
    return cand_sf

def demo(net, image_name, db="./features_sframe.gl", NMS_THRESH_GLOBAL=0.6):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    if isinstance(image_name, basestring):
      im_file = os.path.join(cfg.DATA_DIR, 'demo', 'imgs', image_name)
      im = cv2.imread(im_file)
    else:
      im = image_name.pixel_data 

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    NMS_THRESH = 0.3 # get rid of overlapping windows
    dets_nms_all = np.zeros((0, 5))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind] # class_score for each class, small if the object not present
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # NMS inside each class
        #keep = nms(dets, NMS_THRESH) # reduce redundancy
        #dets = dets[keep, :]
        dets_nms_all = np.vstack((dets_nms_all,  dets)).astype(np.float32)
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
    # Calculate CDF with different threshold
    rois_keep = nms(dets_nms_all, NMS_THRESH_GLOBAL)  # take those with NMS, but might lose some with larger scores due to overlap with another region 
    #rois_keep = dets_nms_all[:, 4].argsort()[::-1][:50]   # take those with maximum score, but might overlap more
    rois_nms = dets_nms_all[rois_keep, :]
    CONF_THRESH=np.linspace(0,1,11)
    cdf = get_cdf(rois_nms, CONF_THRESH)
    rois_sf_withScore = save_img_SF(im, rois_nms)
    rois_sf = rois_sf_withScore.remove_column('score')
    #dfe = gl.feature_engineering.DeepFeatureExtractor('image', model='auto', output_column_prefix=feat)
    alexnet = "~/py-faster-rcnn/tools/alexnet.gl"
    dfe = gl.load_model(alexnet)
    # 28 imgs in the catalogue, calculates c(100)*n(28) = 2800 similarities
    neighbors, db_sf, cand_sf = transform_and_build_nn(rois_sf, dfe, db=db, radius=.6, k=1)
    if "path" in db_sf.column_names():
      db_sf.remove_column('path')
    return neighbors, db_sf, cand_sf

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

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    im_names = ['20134_cols06a_01_PE362778.jpg', '20141_cols30a_01_PE376670.jpg', 'Pasted image at 2016_02_10 09_03 AM.png']
    #im_names = ['bedroom.jpg', 'living_room.jpg', 'kitchen.jpg']
    neigh_all = []
    cand_sf_all = []
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        neighbors, db_sf, cand_sf = demo(net, im_name)
        neigh_all.append(neighbors)
        cand_sf_all.append(cand_sf)
        #image_join(neighbors, db_sf, cand_sf, 9)['image'].show()
