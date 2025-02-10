import abc
import json
import os
import traceback
from collections import defaultdict
from typing import Any

import numpy as np
from einops import rearrange, reduce
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from core import config
from contextlib import contextmanager
import torch
import lightning as L

from dataset_utils.preprocessing_utils import get_coordinates_from_heatmap, renormalise
from dataset_utils.visualisations import plot_heatmaps, plot_heatmaps_and_landmarks_over_img, plot_landmarks_from_img
from utils import metrics
from utils.ema import LitEma
from utils.logging import LogWrapper
from utils.metrics import evaluate_landmark_detection, euclidean_distance


def two_d_softmax(x):
    return torch.softmax(x.flatten(2), dim=-1).view_as(x)


def scale_heatmap_for_plotting(heatmap):
    scale_factor = torch.amax(heatmap, dim=(2, 3), keepdim=True)
    softmax_output = reduce(heatmap / scale_factor * 255, "b c h w -> b 1 h w",
                            "max")
    return softmax_output, scale_factor


class LandmarkDetection(L.LightningModule):
    # Landmark Detection Parent Class
    cfg: config.Config
    batch_idx: int

    def __init__(self, cfg: config.Config, use_ema=True):
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

        self.build_model()

        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=0.9999)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.log_wrapper = None
        self.BCELoss = torch.nn.BCELoss(reduction='sum')

        self.log_train_img_frequency = 10
        self.batch_to_log = 0

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError("Please implement build_model method in child class")

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        batch["x"] = batch["x"].to(device, non_blocking=True)
        batch["y"] = batch["y"].to(device, non_blocking=True)

        batch["y_img"] = batch["y_img"].to(device, non_blocking=True)
        if "y_img_radial" in batch:
            batch["y_img_radial"] = batch["y_img_radial"].to(device, non_blocking=True)
        batch["pixel_size"] = batch["pixel_size"].to(device, non_blocking=True)
        return batch

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
        return batch["y_img"].to(self.device)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # returns loss, loss _dict, output
        # forward should return model output pre activation
        pass

    def calculate_loss(self, output, batch, image):
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

        elif self.cfg.TRAINLOSSES.BCE_WEIGHT > 0:
            activated_heatmap = torch.nn.functional.sigmoid(output)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(output, batch["y_img_radial"], reduction='none')
            bce = torch.mean(torch.sum(bce, dim=(2, 3)))
            loss_dict.update({f'{log_prefix}/combined_bce': bce.item()})
            loss += bce * self.cfg.TRAINLOSSES.BCE_WEIGHT

        elif self.cfg.TRAINLOSSES.NLL_WEIGHT > 0:
            activated_heatmap = two_d_softmax(output)
            nll = -batch["y_img"] * torch.log(activated_heatmap)
            nll = torch.mean(torch.sum(nll, dim=(2, 3)))
            loss_dict.update({f'{log_prefix}/combined_nll': nll.item()})
            loss += nll * self.cfg.TRAINLOSSES.NLL_WEIGHT
        else:
            raise ValueError("No loss function specified")

        # coordinate l2 loss for loss dict
        with torch.no_grad():
            coordinates = get_coordinates_from_heatmap(activated_heatmap,
                                                       k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS).flip(-1)
            l2_coordinate_prediction = torch.mean(euclidean_distance(coordinates, batch["y"].float()))
            loss_dict.update({f'{log_prefix}/l2': l2_coordinate_prediction.item()})
            pixel_sizes = batch["pixel_size"].unsqueeze(1)
            coordinates_scaled = coordinates * pixel_sizes
            real_landmarks_scaled = batch["y"].float() * pixel_sizes
            l2_coordinate_prediction_scaled = torch.mean(euclidean_distance(coordinates_scaled, real_landmarks_scaled))
            loss_dict.update({f'{log_prefix}/l2_scaled': l2_coordinate_prediction_scaled.item()})

            # LOG EXAMPLE IMAGE

            if self.do_img_logging:

                if self.batch_idx == self.batch_to_log and self.current_epoch % self.log_train_img_frequency == 0 or (
                        not self.training and self.batch_idx == self.batch_to_log):
                    # if chosen_class == 0:
                    #     gt = gt_fake
                    # else:
                    #     gt = gt_real
                    with torch.no_grad():
                        # log training samples
                        # log predictions w/ gt, heatmap, heatmap for channel 15, gt for channel 15
                        heatmap_prediction_renorm = activated_heatmap
                        softmax_output, scale_factor = scale_heatmap_for_plotting(heatmap_prediction_renorm)

                        img_pred_log = [
                            plot_landmarks_from_img(renormalise(image, method=self.cfg.DATASET.NORMALISATION),
                                                    heatmap_prediction_renorm,
                                                    true_landmark=batch["y_img_initial"]).cpu().int(),
                            softmax_output.repeat(1, 3, 1, 1).cpu()]
                        random_channel = np.random.randint(0, heatmap_prediction_renorm.shape[1] - 1)
                        img_pred_log.append(
                            (heatmap_prediction_renorm[:, random_channel] / heatmap_prediction_renorm[:,
                                                                            random_channel].max() * 240).unsqueeze(
                                1)
                            .repeat(1, 3, 1, 1).cpu()
                        )
                        # img_pred_log.append(
                        #     (batch["y_img"][:, random_channel] * 255).clamp(0, 255).unsqueeze(1).repeat(1, 3, 1,
                        #                                                                                 1).cpu()
                        # )

                        img_pred_log = self._get_rows_from_list(torch.stack(img_pred_log))

                        try:

                            self.logger.log_image(key=f"Media/{log_prefix}/predictions",
                                                  caption=[
                                                      f"Images {batch['name']} step pixel mre {l2_coordinate_prediction.item():.4f} step mm mre {l2_coordinate_prediction_scaled.item():.4f} scaled by {scale_factor.min().item():.2f}, {scale_factor.max().item():.2f}, {scale_factor.mean().item():.2f}, {scale_factor.std().item():.2f}"],
                                                  images=[img_pred_log])
                        except OSError as e:
                            print(e)
                        del scale_factor, img_pred_log

        return loss, loss_dict, activated_heatmap

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        grid = rearrange(samples, 'n b c h w -> b n c h w')
        grid = rearrange(grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(grid, nrow=n_imgs_per_row)
        return denoise_grid

    def shared_step(self, batch):
        # image is batch["x"], landmark coordinates are batch["y"]
        # landmarks = self.get_landmark_input(batch)
        image = self.get_input(batch, "x")

        output = self(image)
        loss, loss_dict, activated_output = self.calculate_loss(output, batch, image)

        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return loss, loss_dict, activated_output

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

        # if self.current_epoch % 10 == 0 and self.use_ema:
        #     self.model_ema.copy_to(self.model)

        if self.current_epoch % self.log_train_img_frequency == 0:
            self.batch_to_log = np.random.randint(0, self.trainer.num_training_batches)
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
        if self.use_ema:
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
        # Log the learning rate.
        if self.cfg.TRAIN.USE_SCHEDULER != "":
            lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            self.log('Charts/learning_rate', lr, on_epoch=True, on_step=False, prog_bar=False)

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
        with self.ema_scope():
            checkpoint['state_dict'] = self.state_dict()

    def configure_optimizers(self):
        if self.cfg.TRAIN.OPTIMISER.lower() == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                   weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                   betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "adamw":
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                    weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                    betas=(self.cfg.TRAIN.BETA1, self.cfg.TRAIN.BETA2))
        elif self.cfg.TRAIN.OPTIMISER.lower() == "sgd":
            opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg.TRAIN.LR,
                                  weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
                                  momentum=0.9)
        else:
            raise ValueError(f"Optimiser {self.cfg.TRAIN.OPTIMISER} not recognised")
        if self.cfg.TRAIN.USE_SCHEDULER == "multistep":
            # scheduler = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[25, 40], gamma=0.25)]
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 100], gamma=0.1)]
        elif self.cfg.TRAIN.USE_SCHEDULER == "reduce_on_plateau":
            scheduler = [{
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=2,
                                                                        cooldown=2,
                                                                        threshold=0.0001, threshold_mode="abs",
                                                                        verbose=True,
                                                                        min_lr=5e-6),
                "strict": False,
                "monitor": "val/l2_scaled",
            }]
        elif self.cfg.TRAIN.USE_SCHEDULER == "cycle":
            # You only look once config
            base_lr = self.cfg.TRAIN.LR
            max_lr = 0.01
            step_size_up = 20
            step_size_down = 20
            mode = 'triangular2'
            cycle_momentum = False
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                opt, base_lr=base_lr, max_lr=max_lr,
                step_size_up=step_size_up, step_size_down=step_size_down,
                mode=mode, cycle_momentum=cycle_momentum
            )
        else:
            scheduler = []
        return [opt], scheduler
