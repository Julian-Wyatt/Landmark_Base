from abc import ABC
from typing import Any

import torch
from segmentation_models_pytorch import Unet

from trainers.default_trainer import LandmarkDetection


class resnet_unet(LandmarkDetection):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    def build_model(self):
        self.model = Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            # decoder_channels=[512, 384, 256, 128, 64],
            decoder_channels=[256, 256, 256, 128, 64],
            in_channels=self.cfg.DATASET.CHANNELS,
            classes=self.cfg.DATASET.NUMBER_KEY_POINTS,
        )

    def forward(self, x):
        return self.model(x)
