import abc
import json
import os
import traceback
from collections import defaultdict
from typing import Any

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt

from core import config
from contextlib import contextmanager
import torch
import lightning as L

from dataset_utils.preprocessing_utils import get_coordinates_from_heatmap
from dataset_utils.visualisations import plot_heatmaps, plot_heatmaps_and_landmarks_over_img
from utils import metrics
from utils.logging import LogWrapper
from utils.metrics import evaluate_landmark_detection, euclidean_distance


def two_d_softmax(x):
    return torch.softmax(x.flatten(2), dim=-1).view_as(x)


class LandmarkDetection(L.LightningModule):
    # Landmark Detection Parent Class
    cfg: config.Config
    batch_idx: int

    def __init__(self, cfg: config.Config, use_ema=False):
        super(LandmarkDetection, self).__init__()
        self.cfg = cfg
        self.batch_idx = 0
        self.channels = 0
        self.total_annotators = 1
        self.test_coordinates_errors = defaultdict(list)
        self.test_output = dict()
        self.channels = cfg.DATASET.NUMBER_KEY_POINTS

        self.do_img_logging = cfg.TRAIN.LOG_IMAGE
        self.image_size = cfg.DATASET.IMG_SIZE
        self.use_ema = use_ema

        self.log_wrapper = None

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            # if context is not None:
            #     print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                # if context is not None:
                #     print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        if ignore_keys is None:
            ignore_keys = list()
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd,
                                                   strict=False) if not only_model else self.model.load_state_dict(sd)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.shape[-1] in [1, 3]:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def get_landmark_input(self, batch):
        # return dataset.create_landmark_image(batch["y"], self.image_size,
        #                                      eps_window_size=self.cfg.DATASET.LANDMARK_POINT_EPSILON,
        #                                      device=self.device)
        return batch["y_img"].to(self.device)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # returns loss, loss _dict, output
        # forward should return model output pre activation
        pass

    def calculate_loss(self, output, batch):
        # calculate loss here
        loss_dict = dict()
        log_prefix = "train" if self.training else "val"

        if self.device.type != "mps":
            output = output.double()
        loss = torch.tensor(0.0).to(self.device)

        if self.cfg.TRAINLOSSES.NLL_WEIGHT > 0 and self.cfg.TRAINLOSSES.BCE_WEIGHT > 0:
            # TODO: create separate final head for nll and bce
            raise NotImplementedError(
                "Cannot use both NLL and BCE loss simultaneously - To create separate final head for NLL and BCE")

        elif self.cfg.TRAINLOSSES.NLL_WEIGHT > 0:
            activated_heatmap = two_d_softmax(output)
            nll = -batch["y_img"] * torch.log(activated_heatmap)
            nll = torch.mean(torch.sum(nll, dim=(2, 3)))
            loss_dict.update({f'{log_prefix}/combined_nll': nll.detach().item()})
            loss += nll * self.cfg.TRAINLOSSES.NLL_WEIGHT
        elif self.cfg.TRAINLOSSES.BCE_WEIGHT > 0:
            activated_heatmap = torch.nn.functional.sigmoid(output)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(activated_heatmap,
                                                                       batch["y_img_radial"])
            loss_dict.update({f'{log_prefix}/combined_bce': bce.detach().item()})
            loss += bce * self.cfg.TRAINLOSSES.BCE_WEIGHT
        else:
            raise ValueError("No loss function specified")

        # coordinate l2 loss for loss dict
        with torch.no_grad():
            coordinates = get_coordinates_from_heatmap(two_d_softmax(output),
                                                       k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS).flip(-1)
            l2_coordinate_estimate = torch.mean(euclidean_distance(coordinates, batch["y"].float()))
            loss_dict.update({f'{log_prefix}/l2_est': l2_coordinate_estimate.item()})
            pixel_sizes = batch["pixel_size"].unsqueeze(1)
            coordinates_scaled = coordinates * pixel_sizes
            real_landmarks_scaled = batch["y"].float() * pixel_sizes
            l2_coordinate_estimate_scaled = torch.mean(euclidean_distance(coordinates_scaled, real_landmarks_scaled))
            loss_dict.update({f'{log_prefix}/l2_est_scaled': l2_coordinate_estimate_scaled.item()})

        return loss, loss_dict

    def shared_step(self, batch):
        # torch.cuda.empty_cache()
        # image is batch["x"], landmark coordinates are batch["y"]
        # landmarks = self.get_landmark_input(batch)
        image = self.get_input(batch, "x")

        output = self(image)
        loss, loss_dict = self.calculate_loss(output, batch)

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return loss, loss_dict, output

    def training_step(self, batch, batch_idx):

        self.batch_idx = batch_idx
        loss, loss_dict, _ = self.shared_step(batch)

        # self.log_dict(loss_dict, prog_bar=True,
        #               logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def on_validation_start(self) -> None:
        if self.log_wrapper is None:
            self.log_wrapper = LogWrapper(self.cfg, self.logger, lambda x, y, **kwargs: self.log(x, y, **kwargs))

    def on_train_start(self) -> None:
        if self.log_wrapper is None:
            self.log_wrapper = LogWrapper(self.cfg, self.logger, lambda x, y, **kwargs: self.log(x, y, **kwargs))

    def on_test_start(self) -> None:
        if self.log_wrapper is None:
            self.log_wrapper = LogWrapper(self.cfg, self.logger, lambda x, y, **kwargs: self.log(x, y, **kwargs))
        if not os.path.exists(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}"):
            os.makedirs(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}")

    def on_train_epoch_start(self) -> None:
        output_str = f"Epoch {self.current_epoch}"
        if "train/loss" in self.trainer.callback_metrics:
            output_str += f" - Train Loss: {self.trainer.callback_metrics['train/loss']:0.4f}"
        if "val/loss" in self.trainer.callback_metrics:
            output_str += f" - Val Loss: {self.trainer.callback_metrics['val/loss']:0.4f}"
        if "val/l2" in self.trainer.callback_metrics:
            output_str += f" - Val L2: {self.trainer.callback_metrics['val/l2']:0.4f}"
        print(output_str)
        # imgaug.seed(np.random.randint(0, 100000))

    # @abc.abstractmethod
    # def output_to_img_log(self, output, batch, batch_idx=None):
    #     # img_log returns dict with {"heatmaps": torch.Tensor, "video": list of torch.Tensor,
    #     #                            "heatmaps_figure": plt.Figure, "final": plt.Figure}
    #     pass
    def output_to_img_log(self, output, batch, batch_idx=None):
        # img_log returns dict with {"heatmaps": torch.Tensor, "video": list of torch.Tensor,
        #                            "heatmaps_figure": plt.Figure, "final": plt.Figure}
        if not self.cfg.TRAIN.LOG_WHOLE_VAL and batch_idx > 0:
            return None
        with torch.no_grad():
            img_log = {}
            img_log["heatmaps"] = output
            img_log["final"] = plot_heatmaps_and_landmarks_over_img(self.get_input(batch, "x"), output,
                                                                    batch["y"],
                                                                    normalisation_method=self.cfg.DATASET.NORMALISATION)
            img_log["heatmaps_figure"] = plot_heatmaps(output, batch["y"])
        return img_log

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        _, loss_dict_no_ema, output = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema, output = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        img_log = self.output_to_img_log(output, batch, batch_idx)
        if img_log is None:
            return
        # img_log returns dict with {"heatmaps": torch.Tensor, "video": list of torch.Tensor,
        #                            "heatmaps_figure": plt.Figure, "final": plt.Figure}
        log = evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                          pixel_sizes=torch.Tensor([[1, 1]]).to(
                                              img_log["heatmaps"].device),
                                          top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)

        self.log_wrapper.handle_val_logs(batch, batch_idx, log, img_log)

    def on_validation_epoch_end(self) -> None:
        self.log_wrapper.on_val_end()

    @torch.no_grad()
    def unique_test_step(self, batch, batch_idx):
        query_image = self.get_input(batch, "x")
        landmarks = batch["y"]
        with self.ema_scope():
            output = self(query_image)

        img_log = dict()
        img_log["heatmaps"] = output
        img_log["final"] = plot_heatmaps_and_landmarks_over_img(self.get_input(batch, "x"), output,
                                                                landmarks,
                                                                normalisation_method=self.cfg.DATASET.NORMALISATION)
        img_log["heatmaps_figure"] = plot_heatmaps(output, landmarks)

        return img_log

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        img_log = self.unique_test_step(batch, batch_idx)
        self.log_wrapper.handle_test_logs(batch, batch_idx, img_log)

    def on_test_epoch_end(self) -> None:
        # output to csv with format
        # image name, landmark0,...,n average l2, if ddh metrics then angle and line distance
        # columns: image name, landmark0, landmark1, landmark2, landmark3, landmark4, average l2, average angle, average line distance
        # LOGWRAPPER TEST EPOCH END
        self.log_wrapper.on_test_end()

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def on_save_checkpoint(self, checkpoint):
        if self.use_ema:
            with self.ema_scope():
                checkpoint['state_dict'] = self.state_dict()

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMISER.lower() == "adam":
            opt_g = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                     weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                     betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "adamw":
            opt_g = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                      betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "sgd":
            opt_g = torch.optim.SGD(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                    weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                    momentum=0.9)
        else:
            raise ValueError(f"Optimiser {self.cfg.TRAIN.OPTIMISER} not recognised")
        if self.cfg.TRAIN.USE_SCHEDULER == "multistep":
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[25, 40], gamma=0.25)]
        elif self.cfg.TRAIN.USE_SCHEDULER == "reduce_on_plateau":
            scheduler = [{
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=0.5, patience=2,
                                                                        cooldown=2,
                                                                        threshold=0.0001, threshold_mode="abs",
                                                                        verbose=True,
                                                                        min_lr=5e-6),
                "strict": False,
                "monitor": "val/l2_scaled",
            }]
        else:
            scheduler = []
        return [opt_g], scheduler
