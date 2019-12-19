import numpy as np
from skimage import morphology

from helpers import *


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
    

def threshold_labels(Zi, threshold):
    """Convert float predictions into 0 and 1 predictions, using a threshold."""
    return [1 if t >= threshold else 0 for t in Zi]


def road_filters(Zi):
    """Perform image processing to improve the shape of horizontal and vertical roads,
    and to delete noise.
    
    Zi : predicted labels for a single image.
    Returns the updated labels.
    """
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
