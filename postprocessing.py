import numpy as np
from helpers import *

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import data
from skimage import morphology

import matplotlib.pyplot as plt
from matplotlib import cm

def aggregate_labels(labels, patch_size, new_patch_size, threshold):
    """Transforms a linear list of labels into a new list of labels of a bigger patch size.
    
    labels: the list of labels.
    patch_size: the current patch size.
    new_patch_size: the desired output patch size.
    threshold: the percentage of foreground patches needed to classify an
               aggregate as foreground.
    Returns a linear list of labels.
    """
    if new_patch_size % patch_size != 0:
        raise ValueError("The new patch size should be divisible by the old patch size.")
    aggregate_side = int(new_patch_size / patch_size)
    aggregate_n = aggregate_side**2
    new_total_n = int(len(labels) / aggregate_n)
    new_img_side = int(np.sqrt(new_total_n))
    old_img_side = new_img_side * aggregate_side
    new_labels = np.zeros((new_total_n, aggregate_n))
    for k in range(aggregate_side):
        for i in range(new_img_side):
            for j in range(new_img_side):
                for l in range(aggregate_side):
                    idx = (i * aggregate_side + k) * old_img_side + j * aggregate_side + l
                    if labels[idx] == 1:
                        new_labels[i * new_img_side + j, k * aggregate_side + l] = 1
    new_labels = np.sum(new_labels, axis=1)
    res = np.zeros(new_total_n, dtype=np.int8)
    for i, label in enumerate(new_labels):
        if label / aggregate_n >= threshold:
            res[i] = 1
    return res
    

def hough_transform(predicted_img):
    # Classic straight-line Hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(predicted_img, theta=tested_angles)

    # Generating figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(predicted_img, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                 cmap=cm.gray, aspect=1/1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(predicted_img, cmap=cm.gray)
    origin = np.array((0, predicted_img.shape[1]))
    
    # TODO : tune parameters in hough_line_peaks
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=100, min_angle=10, num_peaks=10)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((predicted_img.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    plt.tight_layout()
    plt.show()
    

def clean_labels(img, patch_size):
    """Tries to remove little noise and removes some holes in the lines
    and removes big patches of white like parkings"""
    kernel1 = square(3*patch_size)
    kernel2 = square(9*patch_size)
    kernel3 = np.ones((4*patch_size,1))
    kernel4 = np.ones((1,4*patch_size))

    predicted_img = closing(img, kernel1)
    predicted_img = white_tophat(predicted_img, kernel2)
    predicted_img_vert = opening(predicted_img, kernel3)

    predicted_img_hor = opening(predicted_img, kernel4)

    predict_img_final = cv.addWeighted(predicted_img_vert,1,predicted_img_hor,1,0.0)

    for x in range(len(predict_img_final)):
        for y in range(len(predict_img_final)):
            if predict_img_final[x][y] != 0 :
                predict_img_final[x][y] = 1

    return predict_img_final

def threshold_labels(Zi, threshold):
    return [1 if t >= threshold else 0 for t in Zi]

def road_filters(Zi):
    side = int(np.sqrt(len(Zi)))
    init = label_to_img(side, side, 1, 1, Zi)
    closing = morphology.binary_closing(init, selem=morphology.square(2))
    tophat = morphology.white_tophat(closing, selem=morphology.square(6))
    opening = morphology.opening(tophat, selem=morphology.rectangle(5,1))
    opening_h = morphology.opening(tophat, selem=morphology.rectangle(1,5))
    closing2 = morphology.closing(opening, selem=morphology.rectangle(5,1))
    closing2_h = morphology.closing(opening_h, selem=morphology.rectangle(1,5))
    lab1 = get_labels_from_img(closing2, 0.5, 1)
    lab2 = get_labels_from_img(closing2_h, 0.5, 1)
    lab = [l or lab2[i] for i,l in enumerate(lab1)]
    return lab
