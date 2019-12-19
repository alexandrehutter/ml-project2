import numpy as np
import os,sys

from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

import tensorflow as tf
from tensorflow import keras

from helpers import *
from plots import *
import postprocessing
from models import *

from skimage.filters import gaussian

###########################################
# Initialize parameters :
###########################################

seed = 0

tf.random.set_seed(seed)

patch_size = 16
aggregate_threshold = 0.3

# Extraction function
extraction_func = extract_features_6d

preproc = preprocessing.StandardScaler()

# Using image post-processing
image_proc = True


###########################################
# Data extraction and preprocessing :
###########################################

# Load a set of images
imgs, gt_imgs = load_training_images(n)

# Apply a gaussian blur :
for i in range(len(imgs)) :
    imgs[i] = gaussian(imgs[i], sigma = 2, multichannel = True) 
    
# Extract patches from all images
img_patches = get_patches(imgs, patch_size)
gt_patches = get_patches(gt_imgs, patch_size)

# Get features for each image patch
X = get_features_from_patches(img_patches, extraction_func)
Y = get_labels_from_patches(gt_patches, foreground_threshold)

# Standardization : 
if preproc is not None:
    preproc = preproc.fit(X)
    X = preproc.transform(X)


###########################################
# Select the model :
###########################################

# Uncomment the model that you want to use

# model = knn(X, Y, seed)
model = neural_net(X, Y)



###########################################
# Submission
###########################################

create_submission(model, extraction_func, patch_size, preproc, aggregate_threshold, image_proc)



