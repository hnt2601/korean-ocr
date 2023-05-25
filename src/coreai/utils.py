import re
import base64
import numpy as np
import cv2
import time


def remove_korean_char(list_text):
    text = "".join(list_text)
    output = ""
    for c in text:
        output += c if len(c.encode(encoding="utf_8")) == 1 else ""
    return output


def get_korean_char(list_text):
    text = "".join(list_text)
    output = ""
    for c in text:
        output += c if len(c.encode(encoding="utf_8")) != 1 else ""
    return output


def check_text_all_korea(text):
    len_text = 0
    for c in text:
        if len(c.encode(encoding="utf_8")) != 1:
            len_text += 1
    if len_text == len(text):
        return True
    return False


def split_date_text(list_text):
    text = "".join(list_text)
    date = ""
    issue = ""
    for c in text:
        if len(c.encode(encoding="utf_8")) == 1:
            date += c
        else:
            issue += c
    return date, issue


def convert_box(box):
    return (
        int(box[0][0]),
        min(int(box[0][1]), int(box[1][1])),
        int(box[2][0]),
        max(int(box[2][1]), int(box[3][1])),
    )


def base64_to_cv2(data):
    decode = base64.b64decode(data)
    buffer = np.frombuffer(decode, dtype=np.uint8)
    buffer = buffer.reshape(buffer.shape[0], 1)
    image = cv2.imdecode(buffer, flags=1)

    return image


def cv2_to_base64(image):
    buffer = cv2.imencode(".jpg", image)[1]
    data = base64.b64encode(buffer).decode()

    return data


def get_rotate_crop_image(img, points, pad=3):
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

    # points[0] = points[0] - pad
    # points[2] = points[2] + pad
    # points[1][0] = points[1][0] + pad
    # points[1][1] = points[1][1] - pad
    # points[3][0] = points[3][0] - pad
    # points[3][1] = points[3][1] + pad
    points = np.array(points, dtype=np.float32)
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
    M = cv2.getPerspectiveTransform(points, pts_std)
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


def dilate(ary, N, iterations):
    """Dilate using an NxN '+' sign shape. ary is np.uint8."""
    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[(N - 1) // 2, :] = 1
    dilated_image = cv2.dilate(ary / 255, kernel, iterations=iterations)

    kernel = np.zeros((N, N), dtype=np.uint8)
    kernel[:, int((N - 1) / 2)] = 1
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=iterations)
    return dilated_image


def find_border_components(contours, ary):
    borders = []
    area = ary.shape[0] * ary.shape[1]
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.5 * area:
            borders.append((i, x, y, x + w - 1, y + h - 1))
    return borders


def angle_from_right(deg):
    return min(deg % 90, 90 - (deg % 90))


def remove_border(contour, ary):
    """Remove everything outside a border contour."""
    # Use a rotated rectangle (should be a good approximation of a border).
    # If it's far from a right angle, it's probably two sides of a border and
    # we should use the bounding box instead.
    c_im = np.zeros(ary.shape)
    r = cv2.minAreaRect(contour)
    degs = r[2]
    if angle_from_right(degs) <= 10.0:
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(c_im, [box], 0, 255, -1)
        cv2.drawContours(c_im, [box], 0, 0, 4)
    else:
        x1, y1, x2, y2 = cv2.boundingRect(contour)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(c_im, (x1, y1), (x2, y2), 0, 4)

    return np.minimum(c_im, ary)


def find_components(edges, max_components=16):
    """Dilate the image until there are just a few connected components.
    Returns contours for these components."""
    # Perform increasingly aggressive dilation until there are just a few
    # connected components.
    count = 21
    dilation = 5
    n = 1
    while count > 16:
        n += 1
        dilated_image = dilate(edges, N=3, iterations=n)
        contours, hierarchy = cv2.findContours(
            dilated_image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]
        count = len(contours)
    return contours


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.drawContours(c_im, [c], 0, 255, -1)
        c_info.append(
            {
                "x1": x,
                "y1": y,
                "x2": x + w - 1,
                "y2": y + h - 1,
                "sum": np.sum(ary * (c_im > 0)) / 255,
            }
        )
    return c_info


def split_long_image(im, type_card):
    h, w, c = im.shape
    edges = cv2.Canny(np.asarray(im), 100, 200)
    mask = np.zeros((h, w))
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]
    borders = find_border_components(contours, edges)
    borders.sort(key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))
    border_contour = None
    if len(borders):
        border_contour = contours[borders[0][0]]
        edges = remove_border(border_contour, edges)
    edges = 255 * (edges > 0).astype(np.uint8)
    contours = find_components(edges)

    if len(contours) == 0:
        return None
    c_info = props_for_contours(contours, edges)
    c_info.sort(key=lambda x: -x["sum"])
    for i, c in enumerate(c_info):
        sub_mask = np.ones((h, c["x2"] - c["x1"])) * 255
        mask[0:h, c["x1"] : c["x2"]] = sub_mask
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]
    crops = []
    for ct in contours:
        x1, y1 = ct[0][0][0], ct[0][0][1]
        x2, y2 = ct[2][0][0], ct[2][0][1]
        b_h, b_w = y2 - y1, x2 - x1
        if type_card == "license":
            ratio_pad_h, ratio_pad_w = int(0.1 * b_h), int(0.03 * b_w)
            crop = im.copy()[
                y1 + ratio_pad_h : y2 - ratio_pad_h, x1 + ratio_pad_w : x2 - ratio_pad_w
            ]
        elif type_card == "residencial":
            ratio_pad_h, ratio_pad_w = int(0.15 * b_h), 0
            crop = im.copy()[
                y1 + ratio_pad_h : y2 - ratio_pad_h, x1 + ratio_pad_w : x2 - ratio_pad_w
            ]
        elif type_card == "foreign":
            if b_h < 20 and b_w < 20:
                ratio_pad_h, ratio_pad_w = int(0.2 * b_h), int(0.2 * b_w)
            else:
                ratio_pad_h, ratio_pad_w = 0, 0
            crop = im.copy()[
                y1 - ratio_pad_h : y2 + ratio_pad_h, x1 - ratio_pad_w : x2 + ratio_pad_w
            ]
        crops.append(crop)

    len_c = len(crops)
    pivot = len_c // 2
    if len_c > 2:
        part1 = crops[:pivot]
        part2 = crops[pivot:]
        img1 = np.hstack(part1[::-1])
        img2 = np.hstack(part2[::-1])
        return [img1, img2]
    else:
        img = np.hstack(crops[::-1])
        return [img]


class FaceDetection:
    def __init__(self):
        self.model = cv2.CascadeClassifier(
            "src/coreai/weights/haarcascade_frontalface.xml"
        )

    def __call__(self, image, size=(800, 600)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size)
        faces = self.model.detectMultiScale(gray, 1.3, 5)

        latest_face_cx = None
        latest_area = 0
        for face in faces:
            x, y, w, h = face
            cx = x + w // 2
            area = w * h
            if area > latest_area:
                latest_area = area
                latest_face_cx = cx

        return latest_face_cx


def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def np_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    BArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    try:
        iou = inter / BArea
    except:
        return 0

    return iou


def get_image_overlap(boxes, img, white_pad=6, box_pad=0):
    """
    Sử dụng iou để cắt những text có box đè lên nhau và cắt gọn sát text
    """

    def loop(ov_dict, k, iou, list_out, iou_list):
        if len(ov_dict[k]["ids"]) > 0:
            list_out.append(k)
            iou_list.append(iou)
            for (_id, _iou) in zip(ov_dict[k]["ids"], ov_dict[k]["ious"]):
                loop(ov_dict, _id, _iou, list_out, iou_list)
        else:
            list_out.append(k)
            iou_list.append(iou)

    crop_texts = []
    boxes_sorted = sorted(
        boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True
    )
    temp_boxes = boxes_sorted.copy()
    overlap_boxes = {}
    del_list = []
    for i, t_box in enumerate(temp_boxes):
        overlap_boxes[i] = {"ids": [], "ious": []}
        del_list.append(i)
        for j, box in enumerate(boxes_sorted):
            if j in del_list:
                continue
            iou = np_iou(t_box, box)
            if iou > 0.1:
                overlap_boxes[i]["ids"].append(j)
                overlap_boxes[i]["ious"].append(iou)

    list_images = []
    list_boxes = []
    list_inds = []
    del_keys = []
    for k in overlap_boxes.keys():
        if k in del_keys:
            continue
        new_img = img.copy()
        new_img_h, new_img_w = new_img.shape[:2]
        ids_list = []
        iou_list = []

        if len(overlap_boxes[k]["ids"]) > 0:

            ids_list.append(k)
            iou_list.append(0.0)
            for (_id, _iou) in zip(overlap_boxes[k]["ids"], overlap_boxes[k]["ious"]):
                loop(overlap_boxes, _id, _iou, ids_list, iou_list)
            ids_sort = np.argsort(iou_list)
            if len(ids_sort) > 0:
                ids = np.array(ids_list)[ids_sort][::-1]
                for i, ind in enumerate(ids):
                    del_keys.append(ind)
                    x1, y1, x2, y2 = (
                        boxes_sorted[ind][0],
                        boxes_sorted[ind][1],
                        boxes_sorted[ind][2],
                        boxes_sorted[ind][3],
                    )
                    x1 = np.clip(x1 - box_pad, 0, new_img_w)
                    y1 = np.clip(y1 - box_pad, 0, new_img_h)
                    x2 = np.clip(x2 + box_pad, 0, new_img_w)
                    y2 = np.clip(y2 + box_pad, 0, new_img_h)
                    crop_img = new_img[y1:y2, x1:x2].copy()
                    # cv2.imwrite('/media/hoangnt/Projects/Meditech/korean-ocr/src/coreai/scripts/crop/{}.jpg'.format(time.time()), crop_img)
                    list_inds.append(ind)
                    list_images.append(crop_img)
                    list_boxes.append(boxes_sorted[ind])
                    new_img[y1 + white_pad : y2 - white_pad, x1:x2] = 255
        else:
            del_keys.append(k)
            x1, y1, x2, y2 = (
                boxes_sorted[k][0],
                boxes_sorted[k][1],
                boxes_sorted[k][2],
                boxes_sorted[k][3],
            )
            x1 = np.clip(x1 - box_pad, 0, new_img_w)
            y1 = np.clip(y1 - box_pad, 0, new_img_h)
            x2 = np.clip(x2 + box_pad, 0, new_img_w)
            y2 = np.clip(y2 + box_pad, 0, new_img_h)
            crop_img = img[y1:y2, x1:x2].copy()
            list_images.append(crop_img)
            list_boxes.append(boxes_sorted[k])

    return list_images, list_inds
