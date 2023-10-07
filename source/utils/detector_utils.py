import cv2
import numpy as np
import torch
import torchvision


def preprocess_image(img, target_shape=(736, 512)):
    img = cv2.resize(img, target_shape)
    img = ((img / 255) - 0.5) * 4
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img = np.ascontiguousarray(img)

    return img


def decode_result(datum, threshold=1.0, r=8, iou_threshold=0.7):
    bboxes = {'boxes': [], 'scores': [], 'labels': []}
    datum = {0: datum[:5, :, :],
             1: datum[5:, :, :]}

    for label in [0, 1]:
        mask = (datum[label][0, :, :] >= threshold)

        x_cell = torch.arange(mask.shape[1], device=datum[label].device)
        y_cell = torch.arange(mask.shape[0], device=datum[label].device)

        y_cell, x_cell = torch.meshgrid(y_cell, x_cell)

        x_cell = x_cell[mask]
        y_cell = y_cell[mask]

        x_shift = datum[label][2, :, :][mask]
        y_shift = datum[label][1, :, :][mask]

        x = (x_cell + x_shift) * r
        y = (y_cell + y_shift) * r

        w = datum[label][4, :, :][mask].exp() * r
        h = datum[label][3, :, :][mask].exp() * r

        scores = datum[label][0, :, :][mask]

        for index in range(len(x)):
            bboxes['boxes'].append([x[index] - w[index] / 2,
                                    y[index] - h[index] / 2,
                                    x[index] + w[index] / 2,
                                    y[index] + h[index] / 2])
            bboxes['scores'].append(scores[index])
            bboxes['labels'].append(label)

    bboxes['boxes'] = torch.tensor(bboxes['boxes']).reshape([-1, 4])
    bboxes['scores'] = torch.tensor(bboxes['scores'])
    bboxes['labels'] = torch.tensor(bboxes['labels'])

    to_keep = torchvision.ops.nms(bboxes['boxes'], bboxes['scores'], iou_threshold=iou_threshold)

    bboxes['boxes'] = bboxes['boxes'][to_keep]
    bboxes['scores'] = bboxes['scores'][to_keep]
    bboxes['labels'] = bboxes['labels'][to_keep]

    return bboxes


def decode_batch(batch, threshold=0.1, iou_threshold=0.3):
    res = []
    for index in range(batch.shape[0]):
        res.append(decode_result(batch[index],
                                 threshold=threshold,
                                 iou_threshold=iou_threshold))
    return res


def postprocess(out, thres, iou_thres):
    res = torch.from_numpy(out.copy()).float()
    res[:, [0, 1, 2, 5, 6, 7], :, :] = torch.sigmoid(res[:, [0, 1, 2, 5, 6, 7], :, :])
    bboxes = decode_result(res[0], threshold=thres, iou_threshold=iou_thres)
    return bboxes
