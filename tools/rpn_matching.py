#usr/bin/env python

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
import scipy
import scipy.io as sio
from array import array
import caffe, os, sys, cv2
import argparse
import graphlab as gl
from PIL import Image
from utils import from_pil_image as PIL2gl
#import model_dfe as mdfe
import models.mxnet_dfe as mdfe
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

def transform_cand(cand_sf, qid, dfe):   
    # transform cand offline
    cand_sf = dfe.transform(cand_sf)
    cand_sf = cand_sf.add_row_number()
    cand_sf["qid"] = qid * np.ones((cand_sf.__len__())).astype("int")
    return cand_sf

def transform_and_build_nn(cand_sf, qid, dfe, db="./features_sframe.gl", radius=0.51, k=3):   
    if dfe == "ImageNet_21k":
      cand_sf, top1, top5, top5_prob = mdfe.mx_transform(cand_sf, batch_size=cand_sf.__len__(), dfe="Inception_21k")
    cand_sf = cand_sf.add_row_number()
    if "feature" in cand_sf.column_names():
      cand_sf.rename({"feature": "deep_features.image"})
    cand_sf["qid"] = qid * np.ones((cand_sf.__len__())).astype("int")
    db_sf = gl.SFrame(db)
    db_sf = db_sf.add_row_number()
    if "feature" in db_sf.column_names():
      db_sf.rename({"feature": "deep_features.image"})
    # use label="pid" to include pid in nn
    nn = gl.nearest_neighbors.create(db_sf,label="pid", features=['deep_features.image'],distance='cosine')
    neighbors = nn.query(cand_sf,radius=radius,k=k)
    neighbors_score = neighbors.join(cand_sf, on={"query_label": "id"}, how="inner")
    return neighbors_score, db_sf, cand_sf

def image_join(neighbors, db_sf, cand_sf, query_id):
    """"append the matched catalogue img with a query RoI"""
    tmp_nn = neighbors[neighbors['query_label'] == query_id]
    tmp_db = db_sf.filter_by(tmp_nn['reference_label'], 'pid')
    #name_list = map(lambda x: "db_"+x, tmp_db.column_names())
    #tmp_nn.add_columns([tmp_db], name_list)
    matches = gl.SFrame({"matches": [tmp_db]})
    tmp_nn.add_columns(matches)
    # access: tmp_nn["matches"][0][0-k]["key"]
    matched_db = tmp_db["image"]
    matched_tuple = tmp_nn["image"].append(matched_db)
    return matched_tuple, tmp_nn

def save_img_SF_scale(img, rois, scale=0):
    """save imgs as SFrame"""
    cand_sf = gl.SFrame()
    (H, W, C)= img.shape
    for roi in rois:
        #cropped = img[y:y+h, x:x+w, :]
        (x, y, x1, y1) = roi[0:4]
        h = roi[3] - roi[1]
        w = roi[2] - roi[0]
        if h < 0.25*H:
          y = max(roi[1] - scale/2.0 * h, 0)
          y1 = min(roi[3] + scale/2.0 * h, H)
        if w < 0.25*W:
          x = max(roi[0] - scale/2.0 * w, 0)
          x1 = min(roi[0] + scale/2.0 * w, W)
        cropped = img[y:y1, x:x1, :]
        #cropped = img[roi[1]:roi[3], roi[0]:roi[2], :]
        cropped_img = PIL2gl.from_pil_image(Image.fromarray(cropped))
        #scipy.misc.imsave('crop_%d.jpg'%count, cropped)
        cur_sf = gl.SFrame({'image': [cropped_img], 'score': [roi[4]]})
        cand_sf = cand_sf.append(cur_sf)
    return cand_sf

def save_img_disk(img, rois, im_name=''):
    """save imgs as SFrame"""
    cnt = 0
    for roi in rois:    
        #cropped = img[y:y+h, x:x+w, :]
        cropped = img[roi[1]:roi[3], roi[0]:roi[2], :]
        scipy.misc.imsave("/home/lonestar/py-faster-rcnn/tools/crop_buff/" + im_name + "_CROP%d.jpg"%cnt, cropped)
        cnt += 1

def save_img_SF(img, rois):
    """save imgs as SFrame"""
    cand_sf = gl.SFrame()
    for roi in rois:    
        #cropped = img[y:y+h, x:x+w, :]
        cropped = img[roi[1]:roi[3], roi[0]:roi[2], :]
        cropped_img = PIL2gl.from_pil_image(Image.fromarray(cropped))
        #scipy.misc.imsave('crop_%d.jpg'%count, cropped)
        cand_sf = cand_sf.append(cur_sf)
    return cand_sf

def save_img_array(img, rois):
    """save imgs as nd_array"""
    dim = 299 
    batch_size = rois.shape[0]
    cand_nd = np.zeros((batch_size, dim, dim, 3)) 
    from skimage import io, transform
    cnt = 0
    for roi in rois:    
        #cropped = img[y:y+h, x:x+w, :]
        cropped = img[roi[1]:roi[3], roi[0]:roi[2], :]
        resized_img = transform.resize(cropped, (dim, dim), preserve_range=True)
        cand_nd[cnt, :] = resized_img
        cnt += 1
    return cand_nd

def save_img_array_keep_AR(img, rois, scale=0):
    """save imgs as nd_array, enlarge by scale, croop square, keep aspect ratio"""
    dim = 299
    #dim = 224
    batch_size = rois.shape[0]
    cand_nd = np.zeros((batch_size, dim, dim, 3)) 
    enlarged_rois = np.zeros((batch_size, 4))
    from skimage import io, transform
    cnt = 0
    (H, W, C)= img.shape
    for roi in rois:
        #cropped = img[y:y+h, x:x+w, :]
        (x, y, x1, y1) = roi[0:4]
        h = roi[3] - roi[1]
        w = roi[2] - roi[0]
        if h >= w:
          y = max(roi[1] - scale/2.0 * h, 0)
          y1 = min(roi[3] + scale/2.0 * h, H)
          margin = (y1-y-w)/2.0
          x = max(roi[0] - margin, 0)
          x1 = min(roi[2] + margin, W)
        if w > h:
          x = max(roi[0] - scale/2.0 * w, 0)
          x1 = min(roi[2] + scale/2.0 * w, W)
          margin = (x1-x-h)/2.0
          y = max(roi[1] - margin, 0)
          y1 = min(roi[3] + margin, H)
        enlarged_rois[cnt, :] = np.asarray([x, y, x1, y1])
        cropped = img[y:y1, x:x1, :]
        resized_img = transform.resize(cropped, (dim, dim), preserve_range=True)
        cand_nd[cnt, :] = resized_img
        cnt += 1
    return cand_nd, enlarged_rois

def demo(net, image_name, qid, db="./features_sframe.gl", NMS_THRESH_GLOBAL=0.5, SCORE_THRESH=100):
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
    #rois_keep = nms(dets_nms_all, NMS_THRESH_GLOBAL)  # take those with NMS, but might lose some with larger scores due to overlap with another region 
    #top_keep = dets_nms_all[:, 4].argsort()[::-1][:SCORE_THRESH]   # take those with maximum score, but might overlap more
    top_keep = np.where(dets_nms_all[:, 4] > SCORE_THRESH)[0]
    # TODO: get top_keep by score threshold, say 0.001(130+ rois)
    rois_top = dets_nms_all[top_keep, :]
    nms_keep = nms(rois_top, NMS_THRESH_GLOBAL)
    rois_nms = rois_top[nms_keep, :]
    #CONF_THRESH=np.linspace(0,1,11)
    #cdf = get_cdf(rois_nms, CONF_THRESH)
    #rois_sf_withScore = save_img_SF(im, rois_nms)
    rois_sf = save_img_SF_scale(im, rois_nms, scale=0.5)
    #rois_sf = rois_sf_withScore.remove_column('score')
    #dfe = gl.feature_engineering.DeepFeatureExtractor('image', model='auto', output_column_prefix="deep_features")
    #dfe.save("./alexnet.gl")
    #alexnet = "~/py-faster-rcnn/tools/alexnet.gl"
    #dfe = gl.load_model(alexnet)
    dfe = "ImageNet_21k"
    # For saving crops and features
    #cand_sf = transform_cand(rois_sf, qid, dfe)

    #return cand_sf
    neighbors, db_sf, cand_sf = transform_and_build_nn(rois_sf, qid, dfe, db=db, radius=.7, k=3)
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
