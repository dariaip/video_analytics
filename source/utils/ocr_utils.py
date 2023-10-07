import cv2
import numpy as np
from typing import Tuple


def resize_and_pad(image: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    """
    Resize image maintaining aspect ration and add pads to desired shape.
    Inputs:
    image: np.ndarray - target image
    shape: Tuple[int] - target shape (x,y)
    """

    t_h, t_w = shape  # change x and y
    zeros = np.zeros((t_h, t_w, 3)).astype(np.uint8)
    h, w, _ = image.shape
    if w > h:
        resized = cv2.resize(image.copy(), (t_w, int(t_h * (h / w))))
        y = (zeros.shape[0] - resized.shape[0]) // 2
        zeros[y:y + resized.shape[0], ...] = resized
    elif w < h:
        resized = cv2.resize(image.copy(), (int(t_w * (w / h)), t_h))
        x = (zeros.shape[1] - resized.shape[1]) // 2
        zeros[:, x:x + resized.shape[1], ...] = resized
    else:
        resized = cv2.resize(image.copy(), (t_w, t_h))
        zeros = resized

    return zeros


def preprocess_img_ocr(image, shape=(224, 224)):
    image = resize_and_pad(image, shape)
    image = image.transpose(2, 0, 1)

    image = image / 255
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    return image


def topk(array, k, axis=-1, sorted=True):
    partitioned_ind = (
        np.argpartition(array, -k, axis=axis)
        .take(indices=range(-k, 0), axis=axis)
    )
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)

    if sorted:
        sorted_trunc_ind = np.flip(
            np.argsort(partitioned_scores, axis=axis), axis=axis
        )

        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores

    return scores, ind


def postprocess_ocr(trt_outputs, alphabet='0123456789ABCEHKMOPTXY'):
    confidences, symbols = [i.flatten() for i in topk(trt_outputs[0], 1, axis=1)]

    blank = len(alphabet)
    label, conf_list, buf = '', list(), blank

    for i in range(symbols.shape[0]):
        if symbols[i] == blank or symbols[i] == buf:
            buf = symbols[i]
            continue

        buf = symbols[i]
        label += alphabet[buf]

    return label
