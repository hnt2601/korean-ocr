import os
import sys

import cv2
import copy
import numpy as np
import math
import time

import paddle.fluid as fluid

from . import utility
from .ppocr.logger import initial_logger

logger = initial_logger()
from .ppocr.character import CharacterOps


class Recognizer(object):
    def __init__(self, cfg):
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
        ) = utility.create_predictor(cfg)
        self.use_zero_copy_run = cfg["use_zero_copy_run"]
        self.rec_image_shape = [int(v) for v in cfg["rec_image_shape"].split(",")]
        self.character_type = cfg["rec_char_type"]
        self.rec_batch_num = cfg["rec_batch_num"]
        self.text_len = cfg["max_text_length"]
        self.loss_type = cfg["loss_type"]
        self.char_dict = cfg["rec_char_dict_path"]
        self.use_space_char = cfg["use_space_char"]

        char_ops_params = {
            "character_type": self.character_type,
            "character_dict_path": self.char_dict,
            "use_space_char": self.use_space_char,
            "max_text_length": self.text_len,
            "loss_type": self.loss_type,
        }
        self.char_ops = CharacterOps(char_ops_params)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        wh_ratio = max(max_wh_ratio, imgW * 1.0 / imgH)
        if self.character_type == "ch":
            imgW = int((32 * wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0 : img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length, char_num):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = (
            np.array(range(0, feature_dim)).reshape((feature_dim, 1)).astype("int64")
        )
        gsrm_word_pos = (
            np.array(range(0, max_text_length))
            .reshape((max_text_length, 1))
            .astype("int64")
        )

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length]
        )
        gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1, [1, num_heads, 1, 1]).astype(
            "float32"
        ) * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length]
        )
        gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2, [1, num_heads, 1, 1]).astype(
            "float32"
        ) * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2,
        ]

    def process_image_srn(
        self, img, image_shape, num_heads, max_text_length, char_ops=None
    ):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]
        char_num = char_ops.get_char_num()

        [
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2,
        ] = self.srn_other_inputs(image_shape, num_heads, max_text_length, char_num)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)

        return (
            norm_img,
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2,
        )

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [["", 0.0]] * img_num
        batch_num = self.rec_batch_num
        predict_time = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.loss_type != "srn":
                    norm_img = self.resize_norm_img(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                else:
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]],
                        self.rec_image_shape,
                        8,
                        25,
                        self.char_ops,
                    )
                    encoder_word_pos_list = []
                    gsrm_word_pos_list = []
                    gsrm_slf_attn_bias1_list = []
                    gsrm_slf_attn_bias2_list = []
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])

            norm_img_batch = np.concatenate(norm_img_batch, axis=0)
            norm_img_batch = norm_img_batch.copy()

            if self.loss_type == "srn":
                starttime = time.time()
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(gsrm_slf_attn_bias2_list)
                starttime = time.time()

                norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
                encoder_word_pos_list = fluid.core.PaddleTensor(encoder_word_pos_list)
                gsrm_word_pos_list = fluid.core.PaddleTensor(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = fluid.core.PaddleTensor(
                    gsrm_slf_attn_bias1_list
                )
                gsrm_slf_attn_bias2_list = fluid.core.PaddleTensor(
                    gsrm_slf_attn_bias2_list
                )

                inputs = [
                    norm_img_batch,
                    encoder_word_pos_list,
                    gsrm_slf_attn_bias1_list,
                    gsrm_slf_attn_bias2_list,
                    gsrm_word_pos_list,
                ]

                self.predictor.run(inputs)
            else:
                starttime = time.time()
                if self.use_zero_copy_run:
                    self.input_tensor.copy_from_cpu(norm_img_batch)
                    self.predictor.zero_copy_run()
                else:
                    norm_img_batch = fluid.core.PaddleTensor(norm_img_batch)
                    self.predictor.run([norm_img_batch])

            if self.loss_type == "ctc":
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                rec_idx_lod = self.output_tensors[0].lod()[0]
                predict_batch = self.output_tensors[1].copy_to_cpu()
                predict_lod = self.output_tensors[1].lod()[0]
                elapse = time.time() - starttime
                predict_time += elapse
                for rno in range(len(rec_idx_lod) - 1):
                    beg = rec_idx_lod[rno]
                    end = rec_idx_lod[rno + 1]
                    rec_idx_tmp = rec_idx_batch[beg:end, 0]
                    preds_text = self.char_ops.decode(rec_idx_tmp)
                    beg = predict_lod[rno]
                    end = predict_lod[rno + 1]
                    probs = predict_batch[beg:end, :]
                    ind = np.argmax(probs, axis=1)
                    blank = probs.shape[1]
                    valid_ind = np.where(ind != (blank - 1))[0]
                    if len(valid_ind) == 0:
                        continue
                    score = np.mean(probs[valid_ind, ind[valid_ind]])
                    # rec_res.append([preds_text, score])
                    rec_res[indices[beg_img_no + rno]] = [preds_text, score]
            elif self.loss_type == "srn":
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                probs = self.output_tensors[1].copy_to_cpu()
                char_num = self.char_ops.get_char_num()
                preds = rec_idx_batch.reshape(-1)
                elapse = time.time() - starttime
                predict_time += elapse
                total_preds = preds.copy()
                for ino in range(int(len(rec_idx_batch) / self.text_len)):
                    preds = total_preds[ino * self.text_len : (ino + 1) * self.text_len]
                    ind = np.argmax(probs, axis=1)
                    valid_ind = np.where(preds != int(char_num - 1))[0]
                    if len(valid_ind) == 0:
                        continue
                    score = np.mean(probs[valid_ind, ind[valid_ind]])
                    preds = preds[: valid_ind[-1] + 1]
                    preds_text = self.char_ops.decode(preds)

                    rec_res[indices[beg_img_no + ino]] = preds_text
            else:
                rec_idx_batch = self.output_tensors[0].copy_to_cpu()
                predict_batch = self.output_tensors[1].copy_to_cpu()
                elapse = time.time() - starttime
                predict_time += elapse
                for rno in range(len(rec_idx_batch)):
                    end_pos = np.where(rec_idx_batch[rno, :] == 1)[0]
                    if len(end_pos) <= 1:
                        preds = rec_idx_batch[rno, 1:]
                        score = np.mean(predict_batch[rno, 1:])
                    else:
                        preds = rec_idx_batch[rno, 1 : end_pos[1]]
                        score = np.mean(predict_batch[rno, 1 : end_pos[1]])
                    preds_text = self.char_ops.decode(preds)
                    # rec_res.append([preds_text, score])
                    rec_res[indices[beg_img_no + rno]] = preds_text

        return rec_res
