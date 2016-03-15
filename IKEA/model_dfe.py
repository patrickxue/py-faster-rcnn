import _init_paths
import caffe
import graphlab as gl
import numpy as np
import glob
import os
import pandas as pd
import ipdb

def batch_images(image_path, batch_size):
    images = glob.glob(image_path + '/*jpg')  # used when input arg is a path
    #images = image_path
    for i in xrange(0, len(images), batch_size):
        print "now working on images %d through %d" %(i,i+batch_size)
        yield images[i:i+batch_size]

def extract_labels_features(images, counter, net, transformer, layer="fc7", topk=5):
    #TODO: deal with gl.Image directly instead of reading from disk
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), images)
    out = net.forward()

    idxpreds = np.argsort(-out['prob'], axis=1)
    fc7 = net.blobs[layer]
    labels = []

    for i,j in enumerate(idxpreds):
        labels.append([images[i]])

    labels = np.array(labels)
    #features_with_labels = np.column_stack((labels, fc7.data))
    feat_lab_sf = gl.SFrame({"labels": labels, "features": fc7.data.reshape(labels.shape[0], -1)})
    return feat_lab_sf
    #df = pd.DataFrame(features_with_labels)
    #df.to_csv('%s_ikea_features_with_labels_%d.csv'%(model,counter),index=False,header=False)

def load_model(batchsize):
    caffe.set_mode_gpu()
    # setup net with ( structure definition file ) + ( caffemodel ), in test mode
    net = caffe.Net('./googlenet_places205/deploy_places205.protxt',
                    './googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel', 
                     caffe.TEST)

    #net = caffe.Net('./placesCNN/places205CNN_deploy.prototxt',
    #                './placesCNN/places205CNN_iter_300000.caffemodel', 
    #                 caffe.TEST)

    # add preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
    mean_file = np.array([105.908874512, 114.063842773, 116.282836914]) 
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

#model = 'alexnet_places2'
model = "googlenet_places2"
#layer="pool5/7x7_s1"
counter = 0
failed = []
# dump img
#db_sf = gl.load_sframe("./cata_db_image.gl")
#map(lambda x: scipy.misc.imsave("./cata_db_img/%s.jpg" %x["pid"], x["image"].pixel_data), db_sf)

def dump_img():
    db_sf = gl.load_sframe("./cata_db_image.gl")
    import scipy
    for entry in db_sf:
        img = entry["image"] 
        pid = entry["pid"]
        scipy.io.imsave("./cata_db_img/%s.jpg" %pid) 

def ext_feat(image_path="./crop_buff", layer="fc7", batchsize=25):
    features_all = gl.SFrame()
    net, transformer = load_model(batchsize)
    for images in batch_images(image_path,batchsize):
        global counter
        counter += 1
        try:
            features_sf  = extract_labels_features(images, counter, net, transformer, layer=layer)
            features_all = features_all.append(features_sf)
        except:
            failed.append(images)
            print "failed on counter: %d" %counter
            continue

    return features_all

if __name__ == "__main__":
   global model
   #features_all  = ext_feat(image_path="./cata_db_img")
   # for googlenet_inception model
   model = "googlenet_places2"
   features_all  = ext_feat(image_path="./cata_db_img", layer="pool5/7x7_s1")
   features_all.save("./features_GoogLeNet_PLACES_db.gl")
