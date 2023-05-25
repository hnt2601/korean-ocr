import copy
import json
import os
import re
import time

import cv2
import numpy as np
import requests
from PIL import Image
from src import ocr_logger as logger
from src.config import PADDLE_API

from .form import driver_form, foreign_form, mask, residencial_form
from .ocr import Recognizer
from .preprocess import get_roi
from .utils import (
    FaceDetection,
    base64_to_cv2,
    check_text_all_korea,
    convert_box,
    cv2_to_base64,
    get_image_overlap,
    get_iou,
    get_korean_char,
    get_rotate_crop_image,
    remove_korean_char,
    split_date_text,
    split_long_image,
)


class ContentExtraction:
    def __init__(self):
        self.fd = FaceDetection()
        self.recognizer = Recognizer()
        self.form_scale = (800, 600)

    def detect(self, base64_roi):
        # Detection API
        headers = {"Content-type": "application/json"}
        data = {"feed": [{"image": base64_roi}], "fetch": ["res"]}
        t0 = time.time()
        r = requests.post(url=PADDLE_API, headers=headers, data=json.dumps(data))
        t1 = time.time()
        # print("Time detected: ", t1 - t0)
        detections = r.json()["result"]

        return detections

    def __call__(self, base64_image):
        information = {}
        results = {}
        detections = []

        image = base64_to_cv2(base64_image)
        roi = get_roi(image)
        base64_roi = cv2_to_base64(roi)

        while True:
            detections = self.detect(base64_roi)
            if not detections:
                # fit image
                base64_roi = base64_image
                tmp_img = image.copy()
                detections = self.detect(base64_roi)
                break
            else:
                # not fit image
                image = roi.copy()
                tmp_img = image.copy()
                break

        tmp_boxes = []
        for ret in detections:
            x1, y1, x2, y2 = (
                int(ret[0][0]),
                int(ret[0][1]),
                int(ret[2][0]),
                int(ret[2][1]),
            )
            # cv2.rectangle(tmp_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            tmp_boxes.append([x1, y1, x2, y2])
        # cv2.imwrite("a.jpg", tmp_img)
        overlaps, list_inds = get_image_overlap(tmp_boxes, image)

        latest_face_cx = self.fd(image, self.form_scale)

        if latest_face_cx is None:
            return mask

        # Rescale tất cả đầu vào về cùng tỉ lệ 800x600 để xét ngưỡng vị trí box
        im_h, im_w = image.shape[:2]
        ratio_h, ratio_w = self.form_scale[1] / im_h, self.form_scale[0] / im_w
        resized_img = cv2.resize(image, self.form_scale)
        side = resized_img.shape[1] // 2

        # Mỗi loại giấy tờ có dạng form riêng
        type_card = ""
        type_form = {}
        images = []
        boxes = []

        # Kiểm tra đầu vào là giấy tờ xe hoặc cmnd dựa vào vị trí khuôn mặt
        if 0 < latest_face_cx < side:
            type_card = "license"
            type_form = driver_form
        elif side < latest_face_cx < resized_img.shape[1]:
            type_card = "residencial"
            type_form = residencial_form

        # warp ảnh text nghiêng
        if type_card == "residencial":
            detections = sorted(
                detections,
                key=lambda x: (x[1][0] - x[0][0]) * (x[3][1] - x[0][1]),
                reverse=True,
            )
        rotated_images = [get_rotate_crop_image(tmp_img, box) for box in detections]

        if type_card == "residencial":
            for i, ind in enumerate(list_inds):
                rotated_images[ind] = overlaps[i]

        # convert từ box 4 điểm sang box 2 điểm
        converted_boxes = list(map(lambda x: convert_box(x), detections))
        for rotated_img, box in zip(rotated_images, converted_boxes):
            # cv2.imwrite(f"src/coreai/scripts/crop/{time.time()}.jpg", rotated_img)
            xmin, ymin, xmax, ymax = box
            # Cập nhật toạ độ theo tỉ lệ đã resize
            xmin, ymin, xmax, ymax = (
                int(xmin * ratio_w),
                int(ymin * ratio_h),
                int(xmax * ratio_w),
                int(ymax * ratio_h),
            )
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

            # Tiền xử lí ảnh
            if type_card == "license":
                license_date_xmin = min(
                    type_form["Renewal date"][0], type_form["Valid from"][0]
                )
                license_date_ymin = min(
                    type_form["Renewal date"][1], type_form["Valid from"][1]
                )
                license_date_xmax = max(
                    type_form["Renewal date"][2], type_form["Valid from"][2]
                )
                license_date_tmax = max(
                    type_form["Renewal date"][3], type_form["Valid from"][3]
                )
                if (
                    license_date_xmin < cx < license_date_xmax
                    and license_date_ymin < cy < license_date_tmax
                ):
                    # cắt text dài
                    split_imgs = split_long_image(rotated_img, type_card)
                    if split_imgs is not None:
                        for im in split_imgs:
                            if im.shape[1] < 10:
                                continue
                            # cv2.imwrite(f"src/coreai/scripts/crop/{time.time()}.jpg", im)
                            images.append(im)
                            boxes.append([xmin, ymin, xmax, ymax])
                else:
                    images.append(rotated_img)
                    boxes.append([xmin, ymin, xmax, ymax])

            elif type_card == "residencial":
                residence_addr_xmin = type_form["Address"][0]
                residence_addr_ymin = type_form["Address"][1]
                residence_addr_xmax = type_form["Address"][2]
                residence_addr_ymax = type_form["Address"][3]
                if (
                    residence_addr_xmin < cx < residence_addr_xmax
                    and residence_addr_ymin < cy < residence_addr_ymax
                ):
                    # cắt text dài
                    split_imgs = split_long_image(rotated_img, type_card)
                    if split_imgs is not None:
                        for im in split_imgs:
                            if im.shape[1] < 10:
                                continue
                            # cv2.imwrite(f"src/coreai/scripts/crop/{time.time()}.jpg", im)
                            images.append(im)
                            boxes.append([xmin, ymin, xmax, ymax])
                else:
                    images.append(rotated_img)
                    boxes.append([xmin, ymin, xmax, ymax])
            else:
                images.append(rotated_img)
                boxes.append([xmin, ymin, xmax, ymax])
        # OCR
        t2 = time.time()
        texts = self.recognizer(images)
        t3 = time.time()
        # print("Time predicted: ", t3 - t2)
        # print(texts)

        # Kiểm tra đầu vào có phải cmt nước ngoài hay không?
        # Bắt buộc phải kiểm tra trước điều kiện passport
        if len([x for x in texts if re.search("KOR", x)]) != 0:
            type_card = "foreign"
            type_form = foreign_form
            # images = copy.deepcopy(rotated_images)
            # boxes = copy.deepcopy(converted_boxes)
            # new_images = []
            # new_boxes = []
            # for rotated_img in images:
            #     split_imgs = split_long_image(rotated_img, type_card)
            #     if split_imgs is not None:
            #         for im in split_imgs:
            #             cv2.imwrite(f"src/coreai/scripts/crop/{time.time()}.jpg", im)
            #             new_images.append(im)
            #             new_boxes.append([xmin, ymin, xmax, ymax])

            # images = copy.deepcopy(new_images)
            # boxes = copy.deepcopy(new_boxes)
            # texts = copy.deepcopy(self.recognizer(images))
            # print(texts)

        # Kiểm tra đầu vào có phải hộ chiếu hay không?
        if len([x for x in texts if re.search("PASSPORT", x)]) != 0:
            # Hộ chiếu không có form, không resize tỉ lệ toạ độ, không cắt text
            type_card = "passport"
            type_form = {}
            images = copy.deepcopy(rotated_images)
            boxes = copy.deepcopy(converted_boxes)
            texts = copy.deepcopy(self.recognizer(images))
            # print(texts)

        # Phân tích trường nội dung text dựa vào vị trí đã định nghĩa trước
        if type_form:
            for text, box in zip(texts, boxes):
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                for field, threshold in type_form.items():
                    if (threshold[0] < cx < threshold[2]) and (
                        threshold[1] < cy < threshold[3]
                    ):
                        # Cập nhật text và toạ độ tâm thuộc trường của nó
                        results.setdefault(field, []).append([(cx, cy), text])

        if type_card == "license":
            logger.info(f"Processing license")
            for field, list_info in results.items():
                list_text = [text for ct, text in list_info]
                list_text = list_text[::-1]
                if field == "Address":
                    t = (
                        " ".join(list_text)
                        .replace("루 ", "")
                        .replace("소", "")
                        .replace("주 ", "")
                        .replace("즈 ", "")
                        .lstrip(" ")
                    )
                    information[field] = t
                elif field == "Serial number":
                    information[field] = remove_korean_char(list_text).replace(" ", "")
                elif field == "No":
                    information[field] = remove_korean_char(list_text)
                elif field == "Full name":
                    t = "".join(list_text)
                    t = t[t.find(":") + 1 :]
                    information[field] = (
                        t.replace(" ", "")
                        .replace("性", "")
                        .replace("명", "")
                        .replace("성", "")
                    )
                elif field == "Types":
                    information[field] = list_text
                elif field == "Valid from" or field == "Renewal date":
                    t = remove_korean_char(list_text)
                    t = (
                        t.replace(":", "")
                        .replace(" ", "")
                        .replace(".", "/")
                        .rstrip("/")
                    )
                    t = t[t.find("2") :]
                    information[field] = t
                elif field == "Conditions":
                    information[field] = [
                        (
                            "".join(list_text)
                            .replace("조", "")
                            .replace("건", "")
                            .replace(" ", "")
                            .replace(":", "")
                        )
                    ]
                elif field == "Issuer":
                    date, issue = split_date_text(list_text)
                    information[field] = issue
                    information["Issue date"] = (
                        date.replace(" ", "").replace(".", "/").rstrip("/")
                    )
                else:
                    information[field] = list_text[0]
            information = dict(mask, **information)

        elif type_card == "residencial":
            logger.info(f"Processing residence")
            for field, list_info in results.items():
                list_text = [text for ct, text in list_info]
                if field == "Address":
                    information[field] = (
                        " ".join(list_text).replace(")", "").replace("(", "")
                    )
                elif field == "No":
                    information[field] = "".join(list_text[::-1])
                elif field == "Issuer":
                    information[field] = "".join(list_text[::-1])
                elif field == "Issue date":
                    information[field] = (
                        "".join(list_text[::-1])
                        .replace(" ", "")
                        .replace(".", "/")
                        .rstrip("/")
                    )
                elif field == "Full name":
                    t = " ".join(list_text[::-1])
                    t = (
                        t[: t.find(" ")]
                        .replace("(", "")
                        .replace(")", "")
                        .replace(" ", "")
                    )
                    information[field] = t

            information = dict(mask, **information)

        elif type_card == "foreign":
            logger.info(f"Processing foreign")
            for field, list_info in results.items():
                list_text = [text for ct, text in list_info]
                if field == "No":
                    information[field] = remove_korean_char(list_text[::-1]).lstrip(" ")
                elif field == "Issuer":
                    information[field] = "".join(list_text[::-1]).replace(" ", "")
                elif field == "Issue date":
                    information[field] = (
                        remove_korean_char(list_text[::-1])
                        .replace(".", "/")
                        .lstrip(" ")
                    )
                elif field == "Full name":
                    information[field] = get_korean_char(list_text[::-1])

            information = dict(mask, **information)

        elif type_card == "passport":
            logger.info(f"Processing passport")
            right_side = 0.6 * image.shape[1]
            bot_side = 0.5 * image.shape[0]
            date_dict = {}
            tmp_dict = {}

            for text, box in zip(texts, boxes):
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                if re.search("\d\d \D\D\D \d\d\d\d", text) is not None:
                    date_dict[text] = box
                elif re.search("[A-Z|0-9][0-9][0-9]", text) and len(text) == 9:
                    information["Serial number"] = text
                elif text.isdecimal() and len(text) == 7:
                    information["No"] = text
                elif len(text) == 2:
                    information["Types"] = [text.upper()]
                elif re.search("[A-Z][A-Z][A-Z]", text) and len(text) == 3:
                    information["Address"] = text
                else:
                    tmp_dict[text] = box

            # final update date
            date_dict = dict(
                sorted(date_dict.items(), key=lambda box: (box[1][1] + box[1][3]) // 2)
            )
            list_date = list(date_dict.keys())
            information["Issue date"] = list_date[0]
            information["Renewal date"] = list_date[1] if len(list_date) > 1 else ""
            # final update name
            if list_date[1]:
                date_box = list(date_dict.values())[1]
                for text, other_box in tmp_dict.items():
                    y1 = max(date_box[1], other_box[1])
                    ymin = min(date_box[3] - date_box[1], other_box[3] - other_box[1])
                    y2 = min(date_box[3], other_box[3])
                    inter = max((y2 - y1) / ymin, 0)
                    if inter > 0.6:
                        information["Full name"] = text

            information = dict(mask, **information)
        logger.info(f"[INFO]: {information}")
        return information
