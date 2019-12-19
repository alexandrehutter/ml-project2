import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from helpers import *


def show_img(img, size=5):
    """Shows an image."""
    fig1 = plt.figure(figsize=(size, size))
    plt.imshow(img, cmap='Greys_r')


def show_img_overlay(img, predicted_img, size=10):
    """Shows predictions with a red overlay on top of the base image."""
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    show_img(new_img, size=size)


def show_img_concatenated(img, img2, size=10):
    """Shows two concatenated images (e.g. a base image along with its groundtruth or predictions)."""
    cimg = concatenate_images(img, img2)
    show_img(cimg, size=size)


def show_data_points(X, Y):
    """Plots the data points X (size N,2) with their real or predicted classification Y (size N)."""
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.show()
