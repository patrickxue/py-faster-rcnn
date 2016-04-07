import _init_paths
import caffe
import numpy as np
import glob
import ipdb
import os
import pandas as pd
import graphlab as gl


model = 'alexnet_places2'

def batch_images(image_path, batch_size):
	images = glob.glob(image_path + '/*jpg')
	for i in xrange(0, len(images), batch_size):
		print "now working on images %d through %d" %(i,i+batch_size)
		yield images[i:i+batch_size]

def extract_labels_features(images, net, transformer, counter, tags, skip_tags=False, topk=5):
  net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
  out = net.forward()
  
  idxpreds = np.argsort(-out['prob'], axis=1)
  preds =  -np.sort(-out['prob'], axis=1)[:,:5]
  
  fc7 = net.blobs['fc7']
  
  labels = []
  for i,j in enumerate(idxpreds):
  	if not skip_tags:
  		labels.append([images[i]]+[tags[str(v)] for v in j[:topk]])
  	else:
  		labels.append([images[i]]+['-' for v in j[:topk]])
  
  labels = np.array(labels)
  #features_with_labels = np.column_stack((labels, preds, fc7.data))
  feat_lab_sf = gl.SFrame({"labels": labels, "features": fc7.data.reshape(labels.shape[0], -1)})
  return feat_lab_sf
  #df = pd.DataFrame(features_with_labels)
  #df.to_csv('%s_ikeacrops_features_with_labels_%d.csv'%(model,counter),index=False,header=False)


tags = open('/home/lonestar/IKEA_DATA/IKEA_DATA/placesCNN/categoryIndex_places205.csv')
tags = tags.readlines()
tags = [t.strip().split(' ') for t in tags]
tags = {k:v for v,k in tags}

def load_model(batchsize):
  caffe.set_mode_gpu()
  # setup net with ( structure definition file ) + ( caffemodel ), in test mode
  net = caffe.Net('/home/lonestar/IKEA_DATA/IKEA_DATA/placesCNN/places205CNN_deploy.prototxt',
  			    '/home/lonestar/IKEA_DATA/IKEA_DATA/placesCNN/places205CNN_iter_300000.caffemodel', 
  			    caffe.TEST)
  # add preprocessing
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
  mean_file = np.array([106.908874513, 115.063842773, 116.282836914]) 
  transformer.set_mean('data', mean_file) #### subtract mean ####
  transformer.set_raw_scale('data', 255) # pixel value range
  transformer.set_channel_swap('data', (2,1,0)) 
  if model.split('_')[0] == 'alexnet':
  	net.blobs['data'].reshape(50,3,227,227)
  	
  else: net.blobs['data'].reshape(50,3,224,224)
  # set test batchsize
  data_blob_shape = net.blobs['data'].data.shape
  data_blob_shape = list(data_blob_shape)
  net.blobs['data'].reshape(batchsize, data_blob_shape[1], data_blob_shape[2], data_blob_shape[3])
  return net, transformer

counter = 0

failed = []
def ext_feat(image_path,layer="pool7", batchsize=25):
  net, transformer = load_model(batchsize)
  global counter
  for images in batch_images(image_path, batchsize):
    counter += 1
    ipdb.set_trace()
    try:
      features_sf = extract_labels_features(images, net, transformer, counter, tags, skip_tags=True)
    except:
      failed.append(images)
      print "failed on counter: %d" %counter
      continue
    return features_sf

#image_path = '/home/lonestar/IKEA_DATA/IKEA_DATA/test_img_qid=1'
#batchsize = 10
#ext_feat(image_path, batchsize)
