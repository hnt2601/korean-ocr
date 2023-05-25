import os

import copy
import numpy as np
import sys

import paddle.fluid as fluid

from . import utility
from .ppocr.logger import initial_logger

logger = initial_logger()
from .ppocr.preprocess import DBPreProcess
from .ppocr.postprocess import DBPostProcess
from .ppocr.utils import filter_det


class Detector:
    def __init__(self, cfg):
        self.preprocess_op = DBPreProcess(cfg["det"])
        self.postprocess_op = DBPostProcess(cfg["det"])
        try:
            (
                self.predictor,
                self.input_tensor,
                self.output_tensors,
            ) = utility.create_predictor(cfg["det"])
        except Exception as e:
            logger.info("Error init predictor {}".format(e))

    def __call__(self, img):
        try:
            im, ratios = self.preprocess_op(img)
        except Exception as e:
            logger.info("Error in preprocessing {}".format(e))

        im = im.copy()  # quan trọng vì ảnh hưởng kết quả đầu ra của dự đoán
        im = fluid.core.PaddleTensor(im)
        self.predictor.run([im])
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        try:
            dt_boxes = self.postprocess_op(outputs[0], [ratios])[0]
            dt_boxes = filter_det(dt_boxes, img.shape)
        except Exception as e:
            logger.info("Error in postprocessing {}".format(e))

        return dt_boxes
