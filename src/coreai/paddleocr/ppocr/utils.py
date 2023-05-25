import os
import sys

from .logger import initial_logger

logger = initial_logger()
import base64

import cv2
import numpy as np
from paddle.fluid.core import AnalysisConfig, create_paddle_predictor


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def clip_det(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def filter_det(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new, dtype=np.float32)
    return dt_boxes
