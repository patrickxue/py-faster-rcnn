This directory contains the scripts used for IKEA catalogue matching. It contains two major steps: region/RoI proposal using a trained object detection model by faster rcnn; second, deep feature extraction on the proposed region, and compute similarity with catalogue images.

Object Detection Models: 
  Pascal VOC model: 20 categories, see $FRCNN_ROOT/tools/demo.py for the list of categories
  MS COCO model: 80 categories, see $FRCNN_ROOT/tools/demo_interactive.py for the list of categories. The 80 categories are superset of PASCAL VOC categories, and each category has more images. See MS COCO paper for more details about the dataset.
  ImageNet detection model: 200 basic level (most commonly seen) categories, see $FRCNN_ROOT/tools/demo_interactive.py for the list of categories.

DFE (Deep Feature Engineering) Models:
  DFE models are used for extracting deep features using trained model. We have three different type of DFE models: GLC Model, Caffe Model, MXNet model. 
  GLC Model: AlexNet, embedding dim: 4096
  Caffe Models: AlexNet_ImageNet (4096), AlexNet_PLACES (4096)
  MXNet models: Inception_BN (batch norm), Inception_v3, Inception_21k

  Inception_v3 has more parameters than Inception_BN, the performance is a little better; Inception_21k is trained on the full imagenet dataset by Bing Xu, which outputs the 21k categories (those categories are not mutually exclusive, whereas other models trained on imagenet 1k data are mutually exclusive.) 

IKEA Datasets:
  We have crawled images from IKEA website and saved them as SFrames. 
  Query images: A living scene which contains many objects in it, along with the ground truth objects in the image. 273 images from different living scenes like living room, bedroom, bathroom, hallway...
  Catalogue images: Products image with white background, 6617 catalogue images from different departments.
  see https://docs.google.com/document/d/134v-0WTBSZ8YNfho4JxgtS_qSRKXuHD3Pt2JXP9izx4/edit#bookmark=id.p0kgkybkyvqz for explanation of different SFrames.
