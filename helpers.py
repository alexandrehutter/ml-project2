import matplotlib.image as mpimg
import numpy as np


### Image manipulation ###


def load_image(infilename):
    """Returns an image, given its file path."""
    data = mpimg.imread(infilename)
    return data


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


### Feature extraction ###


def extract_patches(imgs, patch_size=16):
    """Extracts patches from input images."""
    img_patches = [img_crop(img, patch_size, patch_size) for img in imgs]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    return img_patches


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


def extract_img_features(filename, patch_size=16, f=extract_features_2d):
    """Extracts all the features of an image, by splitting it in patches.
    filename: the path to the image.
    patch_size: the side length of a patch (optional).
    f: The feature extraction function (optional).
    Returns a numpy array of features."""
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ f(img_patches[i]) for i in range(len(img_patches))])
    return X
