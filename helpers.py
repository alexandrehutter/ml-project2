import matplotlib.image as mpimg
import numpy as np
import os,sys
import math

from skimage.morphology import  opening, closing, white_tophat
from skimage.morphology import square

import postprocessing
from plots import *

from skimage.filters import gaussian


### Data extraction ###


def load_training_images(n):
    """Loads n training images and the corresponding groundtruth. 
    Returns two lists, (imgs, gt_imgs), where the first contains training images 
    and the second contains groundtruth images."""
    
    root_dir = "Datasets/training/"
    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"
    
    files = os.listdir(image_dir)
    n = min(n, len(files))
    imgs = [load_image(image_dir + files[i]) for i in range(n)]
    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
    return imgs, gt_imgs


def load_image(path):
    """Returns an image, given its file path."""
    img = mpimg.imread(path)
    return img


def get_patches(imgs, patch_size):
    """Extracts patches from a list of images."""
    img_patches = [img_crop(img, patch_size, patch_size) for img in imgs]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    return img_patches


def get_features_from_patches(img_patches, extract_func):
    """Constructs X array from image patches lists."""
    X = np.asarray([ extract_func(img_patches[i]) for i in range(len(img_patches))])
    return X


def get_labels_from_patches(gt_patches, foreground_threshold):
    """Constructs Y array from image patches lists."""
    def value_to_class(v):
        df = np.sum(v)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    return Y


def get_features_from_img(img, extract_func, patch_size):
    """Constructs the X array of a single image, by splitting it into patches."""
    img_patches = img_crop(img, patch_size, patch_size)
    X = get_features_from_patches(img_patches, extract_func)
    return X


def get_labels_from_img(gt_img, foreground_threshold, patch_size):
    """Constructs Y array from image patches lists."""
    gt_patches = img_crop(gt_img, patch_size, patch_size)
    Y = get_labels_from_patches(gt_patches, foreground_threshold)
    return Y


### Image manipulation ###


def img_float_to_uint8(img):
    """Converts the pixels of an image from float to uint8 values."""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """Concatenates an image and its groundtruth."""
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    """Splits an image into patches of the given size, and returns a list of patches."""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


def label_to_img(imgwidth, imgheight, w, h, labels):
    """Transforms a linear array of predictions (0 and 1 values) into an image.
    w and h represent the patch size."""
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def img_to_label(img, patch_size, Zi):
    """Transforms an image into a linear array of 0 and 1 values"""
    
    height = img.shape[0]/patch_size
    width = img.shape[1]/patch_size
    labels = np.zeros(len(Zi), dtype=np.int8)
    
    for i in range(int(height*width)):
        labels[i] = img[math.floor((i*16)/img.shape[1])][(i*16) % img.shape[1]]
    return labels

        



### Feature extraction ###


def extract_features_6d(img):
    """Extracts a 6-dimensional feature consisting of the average and variance of each RGB channel."""
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat


def extract_features_2d(img):
    """Extracts a 2-dimensional feature consisting of the average gray color as well as variance."""
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


def extract_features_12d(img):
    """Extracts a 12-dimensional feature consisting of the average, max, min, and variance of each RGB channel."""
    feat_m = np.mean(img, axis=(0,1))
    feat_min = np.min(img, axis=(0,1))
    feat_max = np.max(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, [feat_v, feat_min, feat_max])
    return feat


def augment_features_6d_with_adjacent(X, side_px, patch_size, n_images):
    """Extends an array of 6D features to include the mean of the mean of adjacent patches, for each channel.
    
    This function assumes an ordered sequence of square images of the same size.
    X: the features array.
    side_px: side number of pixels of the images.
    patch_size: side number of pixels in a patch.
    n_images: the number of images.
    Returns a features array with D=9.
    """
    if side_px % patch_size != 0:
        raise ValueError("Invalid size")
    side_patches = int(side_px / patch_size)
    patches_per_image = side_patches * side_patches
    vec = []
    for n in range(n_images):
        for i in range(patches_per_image):
            # Compute the mean of the three channels, for each adjacent patch
            means = []
            if i % side_patches != 0:
                means.append(X[i - 1, :3])
            if (i + 1) % side_patches != 0:
                means.append(X[i + 1, :3])
            if i >= side_patches:
                means.append(X[i - side_patches, :3])
            if (patches_per_image - 1 - i) >= side_patches:
                means.append(X[i + side_patches, :3])
            m = np.mean(np.array(means), axis=0)
            vec.append(m)
    X = np.hstack((X, np.array(vec).reshape(-1,3)))
    return X


### Predictions analysis ###


def true_positive_rate(Z, Y):
    """Returns the true positive rate, given a set of predictions Z and the true labels Y."""
    Zn = np.where(Z == 1)[0]
    Yn = np.where(Y == 1)[0]
    TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    return TPR


def true_negative_rate(Z, Y):
    """Returns the true negative rate, given a set of predictions Z and the true labels Y."""
    Zn = np.where(Z == 0)[0]
    Yn = np.where(Y == 0)[0]
    TNR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    return TNR


def false_negative_rate(Z, Y):
    """Returns the false negative rate, given a set of predictions Z and the true labels Y."""
    Zn = np.where(Z == 0)[0]
    Yn = np.where(Y == 1)[0]
    FNR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    return FNR


def false_positive_rate(Z, Y):
    """Returns the false positive rate, given a set of predictions Z and the true labels Y."""
    Zn = np.where(Z == 1)[0]
    Yn = np.where(Y == 0)[0]
    FPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    return FPR


def f_score(Z, Y, beta=1):
    """Returns the F-score, given a set of predictions Z and the true labels Y.
    
    beta: the emphasis given to false negatives (recall). Default is the F1-score.
    """
    fb_num = (1 + beta**2) * true_positive_rate(Z, Y)
    fb_den = fb_num + beta**2 * false_negative_rate(Z, Y) + false_positive_rate(Z, Y)
    fb = fb_num / fb_den
    return fb


### Submission ###


def create_submission(model, extraction_func, patch_size, preproc, aggregate_threshold, image_proc):
    """Loads test images, runs predictions on them and creates a submission file."""
    
    dir_t = "Datasets/test_set_images/"
    n_t = 50  # Number of test images
    output_patch_size = 16  # The patch size expected by AICrowd

    with open('Datasets/submission.csv', 'w') as f:
        f.write('id,prediction\n')

        for img_idx in range(1, n_t+1):
            img_path = dir_t + "test_{0}/test_{0}.png".format(img_idx)
            img = load_image(img_path)
            img = gaussian(img, sigma = 2, multichannel = True)
            # Run predictions
            Xi_t = get_features_from_img(img, extraction_func, patch_size)
            if preproc is not None:
                Xi_t = preproc.transform(Xi_t)
            Zi_t = model.predict(Xi_t)
                
            nn_threshold = 0.3
            
            Zi_t = postprocessing.threshold_labels(Zi_t, nn_threshold)
            
            if image_proc:
                Zi_t = postprocessing.road_filters(Zi_t)
                
            if patch_size != output_patch_size:
                Zi_t = postprocessing.aggregate_labels(Zi_t, patch_size, output_patch_size, aggregate_threshold)
            
            # Write predictions to file
            pred_index = 0
            for j in range(0, img.shape[1], output_patch_size):
                for i in range(0, img.shape[0], output_patch_size):
                    f.write("{:03d}_{}_{},{}\n".format(img_idx, j, i, Zi_t[pred_index]))
                    pred_index += 1
