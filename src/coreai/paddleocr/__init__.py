import yaml
from easydict import EasyDict as edict

from .ppocr.logger import initial_logger
from .det import Detector
from .rec import Recognizer

logger = initial_logger()
import time
import cv2
from .utility import get_rotate_crop_image, sorted_boxes
import numpy as np
from PIL import Image
import copy


class TextDetector:
    def __init__(self):
        logger.info("Paddle: Init model")
        with open(
            "src/coreai/paddleocr/config.yaml",
            "r",
        ) as f:
            config = yaml.safe_load(f)
            config = edict(config)
        self.detector = Detector(config)

    def __call__(self, image, pad=0):
        """
        Params:
            - image: cell image
        """
        # image = cv2.resize(image, (800, 600))
        ori_im = image.copy()
        t0 = time.time()
        results = self.detector(image)
        # boxes = sorted_boxes(boxes)
        # images = []
        # for box in boxes:
        #     tmp_box = copy.deepcopy(box)
        #     crop = get_rotate_crop_image(ori_im, tmp_box)
        #     images.append(crop)

        images = []
        boxes = []
        for box in results:
            # tmp_box = copy.deepcopy(box)
            # crop = get_rotate_crop_image(ori_im, tmp_box)
            x1, y1, x2, y2 = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop = ori_im[y1:y2, x1:x2]
            # cv2.imwrite('src/coreai/scripts/out/{}.jpg'.format(time.time()), crop)
            images.append(crop)


class TextRecognizer:
    def __init__(self):
        logger.info("Paddle: Init model")
        with open(
            "src/coreai/paddleocr/config.yaml",
            "r",
        ) as f:
            config = yaml.safe_load(f)
            config = edict(config)
        self.recognizer = Recognizer(config["rec"])

    def __call__(self, image):
        """
        Params:
            - image: cell image
        """
        results = self.recognizer(image)

        return results
