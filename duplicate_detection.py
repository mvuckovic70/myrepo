#------
# init
#------

import tensorflow as tf
import keras

from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model
from keras import optimizers
from keras.layers import GlobalAveragePooling2D
from keras import applications

import numpy as np
from IPython.display import display, HTML


#-----------------------------------------
# iterate through mongo gridfs collection
#-----------------------------------------


import pandas as pd 
import pymongo 

db         = pymongo.MongoClient().local
collection = db.images.files
data       = pd.DataFrame(list(collection.find()))
fname      = (data['filename'])
display(data[1:10])


#------------------------------------------
# parameters for re-shaping of image input
#------------------------------------------

ROWS = 299
COLS = 299


#-------------------------------------------------------------------------
# creates features from images using Inception ResNetV2 pre-trained model
#-------------------------------------------------------------------------

base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(ROWS, COLS, 3))
x          = base_model.output
x          = GlobalAveragePooling2D()(x)
model      = Model(inputs = base_model.input, outputs = x)
model.compile(loss = 'mae', optimizer = optimizers.Adam(), metrics=['accuracy'])
model.summary()


#-----------------------------------------
# converting images to arrays using keras
#-----------------------------------------

import os, os.path
from keras.preprocessing.image import img_to_array, load_img
imgs = []
path = "/media/milos/16349F1A349EFC45/private/loxodon/data"
valid_images = [".jpg",".gif",".png",".tga", "jpeg", 'bmp']

for f in fname:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(img_to_array(load_img(os.path.join(path,f), target_size=(ROWS, COLS))))
print('image loaded')

print(np.shape(imgs))


#-------------------------
# create image embeddings
#-------------------------

image_embeddings = model.predict(preprocess_input(np.array(imgs)), verbose=1)

image_id_to_name = dict()
image_id_emb     = dict()

for i, emb in enumerate(image_embeddings):
    image_id_to_name[data['_id'][i]] = data['filename'][i]
    image_id_emb[data['_id'][i]] = emb


#-------------
# r-tree init
#-------------

from rtree import index
import rtree

p = rtree.index.Property()
p.dimension = 1536
p.dat_extension = 'data'
p.idx_extension = 'index'
rtree_idx = index.Index( properties = p, interleaved=False)


#----------------------
# adding spatial index
#----------------------

idx = 0
for file_idx, emb in enumerate(image_embeddings):
    print(data['_id'][file_idx])
    point = ()
    for x in emb:
        point += (x,x)

    rtree_idx.insert(idx, point)
    idx += 1


#----------------------------------------------------------------------
# calculate distance / probability of each pairs of digital signatures
#----------------------------------------------------------------------

num_nearest = 5
for key, emb in image_id_emb.items():
    print("image: {}:".format(image_id_to_name[key]))
    point = ()
    for x in emb:
        point += (x,x)

    candidates = rtree_idx.nearest(point, num_nearest)
    for r in list(candidates):
        diff = np.subtract(emb, image_embeddings[r])
        dist = np.sum(np.square(diff))
        if dist < 25:
            print("\t{}: duplicate {}%".format(data['filename'][r], (1 - dist/125)*100))

rtree_idx


