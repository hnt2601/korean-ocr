import os, sys
from .ppocr.logger import initial_logger

logger = initial_logger()
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
import cv2
import numpy as np
import base64


def create_predictor(cfg):
    model_file_path = cfg["model_dir"] + "/model"
    params_file_path = cfg["model_dir"] + "/params"
    if not os.path.exists(model_file_path):
        logger.info("not find model file path {}".format(model_file_path))
        sys.exit(0)
    if not os.path.exists(params_file_path):
        logger.info("not find params file path {}".format(params_file_path))
        sys.exit(0)

    config = AnalysisConfig(model_file_path, params_file_path)

    if cfg["use_gpu"]:
        config.enable_use_gpu(cfg["gpu_mem"], cfg["gpu_ind"])
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(1)
        if cfg["enable_mkldnn"]:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

    config.enable_memory_optim()  # Tối ưu bộ nhớ
    config.disable_glog_info()  # ẩn log infer

    if cfg["use_zero_copy_run"]:
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
    else:
        config.switch_use_feed_fetch_ops(True)

    predictor = create_paddle_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_tensor(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_tensor(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors


def get_rotate_crop_image(img, points, pad=0):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """

    points[0] = points[0] - pad
    points[2] = points[2] + pad
    points[1][0] = points[1][0] + pad
    points[1][1] = points[1][1] - pad
    points[3][0] = points[3][0] - pad
    points[3][1] = points[3][1] + pad

    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    dt_boxes = np.array(dt_boxes)
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (
            _boxes[i + 1][0][0] < _boxes[i][0][0]
        ):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def draw_box(img, boxes):
    for box in boxes:
        box = np.reshape(box, [-1, 1, 2]).astype(np.int32)
        image = cv2.polylines(img, [box], True, (255, 0, 0), 2)
    return image
