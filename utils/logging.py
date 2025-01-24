import json
import os
import traceback
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

import core.config
from dataset_utils.preprocessing_utils import get_coordinates_from_heatmap
from utils import metrics
from utils.metrics import success_detection_rates, evaluate_landmark_detection


def save_to_csv(saving_root_dir: str, run_id: str, coordinates_all_batch: list, batch: dict, total_landmarks: int):
    file_path = f"{saving_root_dir}/tmp/{run_id}/test_landmarks.csv"
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            header = ["image file"] + [f"p{i + 1}x,p{i + 1}y" for i in range(total_landmarks)]
            f.write(",".join(header) + "\n")

    for i in range(len(batch["name"])):
        with open(file_path, "a") as f:
            coordinates = coordinates_all_batch[i]
            output = [f"{int(batch['name'][i]):03d}.bmp"] + [str(i) for i in
                                                             coordinates.flatten().tolist()]
            f.write(",".join(output) + "\n")


def print_error_stats(error_name, error_values, is_scaled=False, tostdout=True):
    mean_error = np.mean(error_values)
    std_error = np.std(error_values)
    unit = "mm" if is_scaled else "pixels"
    if "angle" in error_name:
        unit = "degrees"
    if tostdout:
        print(f"{error_name} {mean_error:.3f} +- {std_error:.3f} {unit}")
    return f" {mean_error:.3f} +- {std_error:.3f} {unit} "


def print_sdr_stats(error_name, error_values, tostdout=True, prefix=""):
    if "scaled" in error_name or "x0" in error_name:
        pixel_sizes = [2.0, 2.5, 3.0, 4.0]
        unit = "mm"
    else:
        pixel_sizes = [5, 10, 20, 40]
        unit = "pixels"
    sdr_stats = success_detection_rates(error_values.flatten(), pixel_sizes)

    if tostdout:
        print(
            f"{prefix} test {error_name} sdr stats for {unit} sizes {pixel_sizes} {[f'{sdr:.3f}' for sdr in sdr_stats]}")


class LogWrapper:
    def __init__(self, cfg: core.config.Config, logger, log_fn):
        self.cfg = cfg
        self.logger = logger
        # self.log(f"test/sdr/{dist}", sdr_stats[i])
        self.log = log_fn  # lambda x,y: self.log(x,y)
        self.test_coordinates_errors = defaultdict(list)

    def handle_train_logs(self):
        pass

    def handle_val_logs(self, batch, batch_idx, log, img_log):
        self.batch_idx = batch_idx

        if torch.mean(batch["pixel_size"]) != 1:
            log_scaled = evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                     pixel_sizes=batch["pixel_size"],
                                                     top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
            self.log("val/l2_scaled", np.mean(log_scaled["l2"]), prog_bar=False, logger=True, on_step=False,
                     on_epoch=True)
            log["l2_scaled"] = log_scaled["l2"]
        else:
            log["l2_scaled"] = log["l2"]
        sweep_minimiser = np.mean(log["l2_scaled"])
        self.log("val/l2", np.mean(log["l2"]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/l1", np.mean(log["l1"]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/ere", np.mean(log["ere"]), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("val/sweep_minimiser", sweep_minimiser, prog_bar=False, logger=True, on_step=False,
                 on_epoch=True)

        if self.cfg.TRAIN.LOG_IMAGE and self.batch_idx % 100 == 0:
            if "final" in img_log:
                self.log_image_to_wandb(img_log["final"], "Media/val/Final", ",".join(batch["name"]),
                                        np.mean(log["l2"]))
            # if "heatmaps_figure" in img_log:
            #     self.log_image_to_wandb(img_log["heatmaps_figure"], "Media/val/Heatmaps", ",".join(batch["name"]))
        if "final" in img_log and type(img_log["final"]) is plt.Figure:
            img_log["final"].clf()
            plt.close("all")
        elif "final" in img_log:
            del img_log["final"]
        if "heatmaps_figure" in img_log and type(img_log["heatmaps_figure"]) is plt.Figure:
            img_log["heatmaps_figure"].clf()
            plt.close("all")
        elif "heatmaps_figure" in img_log:
            del img_log["heatmaps_figure"]

        for key in log:
            self.test_coordinates_errors[f"VAL_{key}"].append(log[key])

    def on_val_end(self):

        errors_dict = {key: np.concatenate(val).reshape(-1, self.cfg.DATASET.NUMBER_KEY_POINTS) for key, val in
                       self.test_coordinates_errors.items() if key in ["VAL_l2", "VAL_l2_scaled"]}
        sdr_metric = "VAL_l2"
        if "VAL_l2_scaled" in errors_dict:
            sdr_metric = "VAL_l2_scaled"
        for k in [sdr_metric]:
            if k in self.test_coordinates_errors:
                if "scaled" in k:
                    pixel_sizes = [1, 1.5, 2.0, 2.5, 3.0, 4.0]
                else:
                    pixel_sizes = [5, 10, 20, 40]
                sdrs = metrics.success_detection_rates(errors_dict[k].flatten(), pixel_sizes)

                for i, sdr in enumerate(sdrs):
                    self.log(f"val/sdr_{k[4:]}_{pixel_sizes[i]}", sdr, prog_bar=False, logger=True)

                self.log(f"val/{k[4:]}_std", np.std(errors_dict[k]), prog_bar=False, logger=True)

                self.current_val_2mm_sdr = sdrs[0]
        self.current_val_mre = np.mean(errors_dict[sdr_metric])

        self.test_coordinates_errors = defaultdict(list)

    def handle_test_logs(self, batch, batch_idx, img_log):

        if self.cfg.TRAIN.LOG_TEST_METRICS:

            log = metrics.evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                      ddh_metrics=self.cfg.DATASET.LOG_DDH_METRICS,
                                                      pixel_sizes=torch.Tensor([[1, 1]]).to(
                                                          img_log["heatmaps"].device),
                                                      top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
            log_scaled = metrics.evaluate_landmark_detection(img_log["heatmaps"], batch["y"],
                                                             ddh_metrics=self.cfg.DATASET.LOG_DDH_METRICS,
                                                             pixel_sizes=batch["pixel_size"],
                                                             top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)

            if batch["landmarks_per_annotator"].shape[1] > 1:

                for annotator in range(batch["landmarks_per_annotator"].shape[1]):
                    annotations = batch["landmarks_per_annotator"][:, annotator, :, :]
                    log_annotator = metrics.evaluate_landmark_detection(img_log["heatmaps"], annotations,
                                                                        ddh_metrics=False,
                                                                        pixel_sizes=batch["pixel_size"],
                                                                        top_k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS)
                    self.log(f"test/l2_annotator_{annotator}", np.mean(log_annotator["l2"]), prog_bar=False,
                             logger=True,
                             on_step=False, on_epoch=True)
                    self.test_coordinates_errors[f"l2_scaled_annotator_{annotator}"].append(log_annotator["l2"])

            # if not os.path.exists(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}"):
            #     os.makedirs(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}")

            for i in range(len(batch["name"])):
                msg = f"Image: {batch['name'][i]}\t"
                for radial_error in log_scaled["l2"][i]:
                    msg += f"\t{radial_error:06.3f} mm"
                msg += f"\taverage: {np.mean(log_scaled['l2'][i]):06.3f} mm"
                print(msg)

                # with open(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/test_results.csv",
                #           "a") as f:
                #     output = [batch['name'][i], *[str(x) for x in log["l2"][i]], str(np.mean(log['l2'][i]))]
                #     if self.cfg.DATASET.LOG_DDH_METRICS:
                #         output += [str(np.mean(log["line_dist"][i]))]
                #         output += [str(np.mean(log["angle_dist"][i]))]
                #
                #     f.write(",".join(output) + "\n")

            for k, v in log.items():
                self.test_coordinates_errors[k].append(v)
                self.log(f"test/{k}", np.mean(v), prog_bar=False, logger=True, on_step=True, on_epoch=True)

            for k, v in log_scaled.items():
                self.test_coordinates_errors[f"{k}_scaled"].append(v)
                if "l2" in k:
                    self.log(f"test/{k}_scaled", np.mean(v), prog_bar=False, logger=True, on_step=True,
                             on_epoch=True)

        coordinates = get_coordinates_from_heatmap(img_log["heatmaps"],
                                                   k=self.cfg.TRAIN.TOP_K_HOTTEST_POINTS).flip(-1)
        l2_coordinate_estimate = torch.mean(metrics.euclidean_distance(coordinates, batch["y"].float()))
        if not self.cfg.TRAIN.LOG_TEST_METRICS:
            self.log(f'test/l2', l2_coordinate_estimate.item(), prog_bar=False, logger=True, on_step=True,
                     on_epoch=True)
        pixel_sizes = batch["pixel_size"].unsqueeze(1)
        coordinates_scaled = coordinates * pixel_sizes
        real_landmarks_scaled = batch["y"].float() * pixel_sizes
        l2_coordinate_estimate_scaled = torch.mean(
            metrics.euclidean_distance(coordinates_scaled, real_landmarks_scaled))
        if not self.cfg.TRAIN.LOG_TEST_METRICS:
            self.log(f'test/l2_scaled', l2_coordinate_estimate_scaled.item(), prog_bar=False, logger=True,
                     on_step=True, )

        if self.cfg.TRAIN.LOG_IMAGE and "final" in img_log:
            self.log_image_to_wandb(img_log["final"], "Media/test/Overlay", ",".join(batch["name"]),
                                    l2_coordinate_estimate_scaled.item())

    def on_test_end(self):
        # if self.cfg.TRAIN.LOG_TEST_METRICS:

        # log test metric dict
        print({key: np.concatenate(val).shape for key, val in self.test_coordinates_errors.items()})

        error_metrics = ['l2', "l2_scaled", 'l1', 'ere']

        errors_dict = {key: np.concatenate(val).reshape(-1, self.cfg.DATASET.NUMBER_KEY_POINTS) for key, val in
                       self.test_coordinates_errors.items() if key in error_metrics}
        with open(f"{self.cfg.TRAIN.SAVING_ROOT_DIR}/tmp/{self.logger.experiment.id}/summary_test_results.csv",
                  "a") as f:
            output = [np.mean(errors_dict["l2"]),
                      *[success_detection_rates(errors_dict["l2"].flatten(), [5, 10, 20, 40])]]
            f.write(",".join([str(x) for x in output]) + "\n")
        self.logger.log_hyperparams({"test_csv": f"{self.logger.root_dir}/test_results.csv"})
        self.logger.log_hyperparams({"summary_test_csv": f"{self.logger.root_dir}/summary_test_results.csv"})
        print("-----------------------------------")
        print("Test Results")
        print("-----------------------------------")
        print("Channel wise mean errors")

        # change to output landmark detail all in one line and new line for new metric []
        # average l2 error for each landmark
        try:
            outputs = ["" for _ in range(len(errors_dict.keys()))]
            for i in range(self.cfg.DATASET.NUMBER_KEY_POINTS):

                for j, val in enumerate(errors_dict.items()):
                    keys, values = val
                    # self.log(f"test/landmark_{i + 1}_{keys}", np.mean(values[:, i]))
                    outputs[j] += print_error_stats(keys, values[:, i], "scaled" in keys or "x0" in keys,
                                                    tostdout=False)

            for j, val in enumerate(errors_dict.items()):
                keys, values = val
                print(f"{keys} {outputs[j]}")

            for keys in ["l2", "l2_scaled", "l2_x0"]:
                if keys in errors_dict:
                    print(f"{keys} SDR stats")
                    for i in range(self.cfg.DATASET.NUMBER_KEY_POINTS):
                        print_sdr_stats(keys, errors_dict[keys][:, i], prefix=f"Landmark {i}")
        except Exception as e:
            print(e, traceback.format_exc())

        print("\n-----------------------------------")

        print("Overall mean errors")

        for error in error_metrics:
            print_error_stats(error, errors_dict[error], "scaled" in error or "x0" in error)

        print("-----------------------------------")

        print_sdr_stats('l2', errors_dict['l2'])

        print("-----------------------------------")

        for error in ['l2_scaled']:
            print_sdr_stats(error, errors_dict[error])
        sdr_stats = success_detection_rates(errors_dict["l2_scaled"].flatten(), [2.0, 2.5, 3.0, 4.0])
        for i, dist in enumerate([2, 2.5, 3, 4]):
            self.log(f"test/sdr/{dist}", sdr_stats[i])

        print("-----------------------------------")

    def log_image_to_wandb(self, figure: plt.Figure, log_name, filename, mre=None):
        import wandb
        if not os.path.exists(f"./{self.logger.experiment.dir}/images"):
            os.makedirs(f"./{self.logger.experiment.dir}/images")

        figure.savefig(f"./{self.logger.experiment.dir}/images/{filename}.png")

        if not log_name.lower().startswith("media"):
            log_name = f"Media/{log_name}"
        self.logger.experiment.log(
            {f"{log_name}": wandb.Image(f"./{self.logger.experiment.dir}/images/{filename}.png",
                                        caption=filename + f"| MRE: {mre:.4f}")})

        plt.clf()
        plt.close("all")
