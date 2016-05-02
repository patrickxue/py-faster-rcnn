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
import urllib2
import rpn_matching as matching
import models.mxnet_dfe as mdfe 


devkit_path = '/home/lonestar/rcnn_finetune/py-faster-rcnn/data/imagenet/ILSVRC2014_devkit' 
synsets = sio.loadmat(os.path.join(devkit_path, 'data', 'meta_det.mat'))
CLASSES = ('__background__',)
for i in xrange(200):
    CLASSES = CLASSES + (synsets['synsets'][0][i][2][0],)

CLASSES = '__background__/person/bicycle/car/motorcycle/airplane/bus/train/truck/boat/traffic light/fire hydrant/stop sign/parking meter/bench/bird/cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe/backpack/umbrella/handbag/tie/suitcase/frisbee/skis/snowboard/sports ball/kite/baseball bat/baseball glove/skateboard/surfboard/tennis racket/bottle/wine glass/cup/fork/knife/spoon/bowl/banana/apple/sandwich/orange/broccoli/carrot/hot dog/pizza/donut/cake/chair/couch/potted plant/bed/dining table/toilet/tv/laptop/mouse/remote/keyboard/cell phone/microwave/oven/toaster/sink/refrigerator/book/clock/vase/scissors/teddy bear/hair drier/toothbrush'.split('/')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def save_cropped(im, class_name, dets_nms_all, im_name, thresh=0.5):
    """Draw detected bounding boxes."""
    rois_nms = dets_nms_all[dets_nms_all[:, 4] > thresh]
    #rois_sf = matching.save_img_SF(im, rois_nms)
    path = im_name + '_CLASS=' + class_name
    if rois_nms.shape[0] == 0:
        #print "no detected objects above threshold" 
        return
    matching.save_img_disk(im, rois_nms, path)
    rois_sf, enlarged_rois = matching.save_img_array_keep_AR(im, rois_nms)
    matching.save_img_disk(im, enlarged_rois, path + '_enlarged')

def vis_detections(im, class_name, dets_nms_all, thresh=0.5):
    """Draw detected bounding boxes."""
    rois_nms = dets_nms_all[dets_nms_all[:, 4] > thresh]
    #rois_sf = matching.save_img_SF(im, rois_nms)
    if rois_nms.shape[0] == 0:
        #print "no detected objects above threshold" 
        return
    rois_sf, enlarged_rois = matching.save_img_array_keep_AR(im, rois_nms)
    features, top1, top5, prob = mdfe.mx_transform(rois_sf, batch_size = rois_sf.__len__())
    rois_sf_old = matching.save_img_array(im, rois_nms)
    features_old, top1_old, top5_old, prob_old = mdfe.mx_transform(rois_sf_old, batch_size = rois_sf_old.__len__())
    inds = np.where(dets_nms_all[:, -1] >= thresh)[0]

    im = im[:, :, (2, 1, 0)]
    cnt = 0
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets_nms_all[i, :4]
        score = dets_nms_all[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}:{:.3f} {:s}:{:.2f} {:s}:{:.2f} {:s}:{:.2f} {:s}:{:.2f} {:s}:{:.2f}'.format(class_name, score, top5_old[cnt][0], prob_old[cnt][0], top5_old[cnt][1], prob_old[cnt][1], top5_old[cnt][2],prob_old[cnt][2], top5_old[cnt][3], prob_old[cnt][3], top5_old[cnt][4],prob_old[cnt][4]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        # put bbox on enlarged area
        bbox_enlarged = enlarged_rois[i, :]
        ax.add_patch(
            plt.Rectangle((bbox_enlarged[0], bbox_enlarged[1]),
                          bbox_enlarged[2] - bbox_enlarged[0],
                          bbox_enlarged[3] - bbox_enlarged[1], fill=False,
                          edgecolor='black', linewidth=3.5)
            )
        ax.text(bbox_enlarged[0], bbox_enlarged[3] + 5,
                '{:s}:{:.2f} {:s}:{:.2f} {:s}:{:.2f} {:s}:{:.2f} {:s}:{:.2f}'.format(top5[cnt][0], prob[cnt][0], top5[cnt][1], prob[cnt][1], top5[cnt][2],prob[cnt][2], top5[cnt][3], prob[cnt][3], top5[cnt][4],prob[cnt][4]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        cnt += 1

    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(top5[cnt], class_name,
    #                                              thresh),
    #              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    im_name = image_name.split('/')[-1].split('.')[0]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    dets_nms_all = np.zeros((0, 5))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #save_cropped(im, cls, dets, im_name, thresh=CONF_THRESH)
        #dets_nms_all = np.vstack((dets_nms_all,  dets)).astype(np.float32)
        
    #vis_detections(im, cls, dets_nms_all, thresh=CONF_THRESH)

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

    prototxt = '/home/haijieg/py-faster-rcnn/models/coco/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = '/home/haijieg/py-faster-rcnn/data/faster_rcnn_models/coco_vgg16_faster_rcnn_final.caffemodel'

    #imagenet model
    #prototxt = '/home/lonestar/rcnn_finetune/py-faster-rcnn/models/VGG16/faster_rcnn_end2end/test.prototxt' 
    #caffemodel = '/home/lonestar/rcnn_finetune/py-faster-rcnn/output/faster_rcnn_end2end/val1/vgg16_faster_rcnn_iter_100000.caffemodel'
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

    scripps_image_list = os.listdir('/data/scripps/raw')
    
    # save img crops
    #cnt = 0
    #for image in scripps_image_list:
    #    imname = '/data/scripps/raw/' + image
    #    print "processing {:d} img: {:s}".format(cnt, image)
    #    demo(net, imname)
    #    cnt += 1

    # interactive demo
    while True:
        url = raw_input("Input image url or ID of scripps network query image: >>>")
        #url = '100'
        if url is 'q':
            break

        if url.startswith('http'):
            print "Opening: ", url
            imname = './tmp.jpg'
            response = urllib2.urlopen(url)
            with open('./tmp.jpg', 'wb') as f:
                f.write(response.read())
        else:
            try:
                image_id = int(url)
                imname = '/data/scripps/raw/'+scripps_image_list[image_id]
                print "Opening: ", imname
            except:
                print "Bad image id or url: %s" % url 
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        demo(net, imname)
        plt.show()
