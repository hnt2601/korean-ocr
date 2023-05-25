import torch
from torch import nn
from torch.nn import functional as F

from .attention_recognition_head import AttentionRecognitionHead
from .resnet_aster import ResNet_ASTER
from .stn_head import STNHead
from .tps_spatial_transformer import TPSSpatialTransformer


class ModelBuilder(nn.Module):
    """
    This is the integrated model.
    """

    def __init__(self, cfg):
        super(ModelBuilder, self).__init__()
        self.cfg = cfg
        self.arch = self.cfg["arch"]
        self.rec_num_classes = self.cfg["len_voc"]
        self.sDim = self.cfg["decoder_sdim"]
        self.attDim = self.cfg["attDim"]
        self.max_len_labels = self.cfg["max_len"]
        self.eos = self.cfg["eos"]
        if self.eos == -1:
            print("ERROR: eos must be greater than -1")
        self.STN_ON = self.cfg["STN_ON"]
        self.tps_inputsize = self.cfg["tps_inputsize"]

        self.encoder = ResNet_ASTER(
            with_lstm=self.cfg["with_lstm"], n_group=self.cfg["n_group"]
        )
        encoder_out_planes = self.encoder.out_planes

        self.decoder = AttentionRecognitionHead(
            num_classes=self.rec_num_classes,
            in_planes=encoder_out_planes,
            sDim=self.sDim,
            attDim=self.attDim,
            max_len_labels=self.max_len_labels,
        )

        if self.STN_ON:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(self.cfg["tps_outputsize"]),
                num_control_points=self.cfg["num_control_points"],
                margins=tuple(self.cfg["tps_margins"]),
            )
            self.stn_head = STNHead(
                in_planes=3,
                num_ctrlpoints=self.cfg["num_control_points"],
                activation=self.cfg["stn_activation"],
            )

    def forward(self, x):
        #         print('input: ', x.shape)
        # rectification
        if self.STN_ON:
            # input images are downsampled before being fed into stn_head.
            # n, 3, 32, 100 => n, 3, 32, 64
            stn_input = F.interpolate(
                x, self.tps_inputsize, mode="bilinear", align_corners=True
            )
            #             print('stn_input ', stn_input.shape)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            #             print('stn_img_feat: ', stn_img_feat.shape, ctrl_points.shape)
            x, _ = self.tps(x, ctrl_points)
        #             print('rectification x: ', x.shape)

        # n, 3, 32, 100 => n, 3, len_text, 512
        encoder_feats = self.encoder(x)
        #         print('encoder_feats: ', encoder_feats.shape)
        rec_pred, rec_pred_scores = self.decoder.beam_search(
            encoder_feats, self.cfg["beam_width"], self.eos
        )  # id of chars
        # n, 3, len_text, 512 => n, max_len_text
        #         print('rec_pred: ', rec_pred.shape)
        return rec_pred, rec_pred_scores
