import numpy as np


def compute_micro_iou(image, gt_image, label_values):
    """
    Compute intersection over union metric for given semantic image and
    semantic GT image.

    Args
        image           :   semantic RGB image [w, h, 3]
        gt_image        :   semantic GT RGB image [w, h, 3]
        label_values    :   RGB value per semantic class

    Returns:
        NP array of IOU values, one for each class label.
        Returns NAN IOU if there union and intersection are both zero.

    """
    iou = []
    image_arr = image.reshape(-1, 3)
    gt_image_arr = gt_image.reshape(-1, 3)

    for label_rgb in label_values:

        image_pixels = np.all(image_arr == label_rgb, axis=-1)
        gt_pixels = np.all(gt_image_arr == label_rgb, axis=-1)

        image_mask = np.zeros((image_arr.shape[0], 1), dtype=np.bool)
        image_mask[np.where(image_pixels)] = True
        gt_mask = np.zeros((image_arr.shape[0], 1), dtype=np.bool)
        gt_mask[np.where(gt_pixels)] = True

        intersection = image_mask * gt_mask
        union = image_mask + gt_mask

        if np.sum(union) > 0:
            iou.append(intersection.sum() / union.sum())
        elif np.sum(intersection) > 0:
            iou.append(0)
        else:
            iou.append(np.nan)

    return np.array(iou)


def rgb_to_onehot(img, labels):
    probability = np.zeros([img.shape[0], img.shape[1], len(labels)])
    for label in labels:
        coords = np.where(np.all(img == np.array(label.color), axis=2))
        one_hot = np.zeros(len(labels))
        one_hot[label.id] = 1
        probability[coords[0], coords[1], :] = one_hot
    return probability


def rgb_to_idx(img, labels):
    idx = np.zeros([img.shape[0], img.shape[1]])
    for label in labels:
        coords = np.where(np.all(img == np.array(label.color), axis=2))
        idx[coords[0], coords[1]] = label.id
    return idx


def outputs_to_rgb(outputs, idx_to_color):
    rgb_ims = [idx_to_color[im] for im in outputs]
    return rgb_ims
