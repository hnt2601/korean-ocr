import torch
from torchvision import transforms
from PIL import Image
import os.path as osp
import numpy as np
import cv2


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath, map_location="cpu")
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = Image.fromarray(img)
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class TransformData(object):
    def __init__(self, imgH=32, imgW=256, keep_ratio=True, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, images):
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                h, w = image.shape[:2]
                ratios.append(w / float(h))
                ratios.sort()
                max_ratio = ratios[-1]
                imgW = int(np.floor(max_ratio * imgH))
                imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
                imgW = min(imgW, 400)

        transform = ResizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        b_images = torch.stack(images)

        return b_images


def get_vocabulary(EOS="EOS", PADDING="PADDING", UNKNOWN="UNKNOWN"):
    voc = []
    with open(
        "src/coreai/ocr/voc.txt",
        "r",
        encoding="utf-8",
    ) as f:
        lines = f.readlines()
        for line in lines:
            char = line.rstrip("\n")
            voc.append(char)

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)
    return voc


def postprocess(pred, id2char, char2id):
    def to_numpy(tensor):
        if torch.is_tensor(tensor):
            return tensor.cpu().numpy()
        elif type(tensor).__module__ != "numpy":
            raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
        return tensor

    # label_seq
    assert pred.dim() == 2

    end_label = char2id["EOS"]
    unknown_label = char2id["UNKNOWN"]
    num_samples, max_len_labels = pred.size()
    num_classes = len(char2id.keys())
    output = to_numpy(pred)

    # list of char list
    pred_list, targ_list = [], []
    for i in range(num_samples):
        pred_list_i = []
        for j in range(max_len_labels):
            if output[i, j] != end_label:
                if output[i, j] != unknown_label:
                    pred_list_i.append(id2char[output[i, j]])
            else:
                break
        pred_list.append(pred_list_i)
    final_text = ["".join(pred) for pred in pred_list]

    return final_text
