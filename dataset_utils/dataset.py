import json
import os
import time
from collections import defaultdict
import random

import cv2
import numpy as np
import pandas as pd
import skimage
import torch
from einops import rearrange, reduce
from imgaug import KeypointsOnImage
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as iaa

import utils.device
from core import config
from core.config import Config
from dataset_utils.dataset_caching import cache_data
from dataset_utils.preprocessing_utils import normalise, simulate_x_ray_artefacts, create_landmark_image, renormalise, \
    create_radial_mask

import lightning as L
import albumentations as A

from dataset_utils.visualisations import plot_landmarks_from_img


class LandmarkDataset(Dataset):

    def __init__(self, cfg: Config, partition="training", augment=False):

        # handle directories
        self.CACHE_DIR = cfg.DATASET.CACHE_DIR
        self.DATASET_NAME = f"{cfg.DATASET.NAME}/{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}"
        self.root_dir = cfg.DATASET.ROOT_DIR
        self.tensor_device = utils.device.get_device()
        self.USE_GRAYSCALE = cfg.DATASET.CHANNELS == 1

        # if type(cfg.DATASET.LABEL_DIR) is str:
        #     if cfg.DATASET.LABEL_DIR.endswith(".csv"):
        #         self.annotation_dirs = [""]
        #         self.annotations_dataframe = pd.read_csv(
        #             f"{self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{cfg.DATASET.LABEL_DIR}")
        #     else:
        #         self.annotation_dirs = [cfg.DATASET.LABEL_DIR]
        # else:
        #     self.annotation_dirs = cfg.DATASET.LABEL_DIR

        self.total_landmarks = cfg.DATASET.NUMBER_KEY_POINTS
        self.dataset_pixels_per_mm = cfg.DATASET.PIXELS_PER_MM

        # handle partitions
        self.partition = partition
        self.partition_file = f"{cfg.DATASET.ROOT_DIR}{'/' if cfg.DATASET.ROOT_DIR[-1] != '/' else ''}"
        if cfg.DATASET.PARTITION_FILE == "":
            self.partition_file += "partitions/default.json"
        else:
            self.partition_file += cfg.DATASET.PARTITION_FILE

        self.annotations = []
        self.images = []
        self.metas = []
        self.handle_partition()

        # perform augmentation
        self.augment = augment
        self.IMG_SIZE = cfg.DATASET.IMG_SIZE

        self.SIGMAS = np.ones(self.total_landmarks) * cfg.DATASET.GT_SIGMA

        self.USE_GAUSSIAN = cfg.DATASET.GT_SIGMA > 0

        self.scale_transform_skew = iaa.Affine(
            scale={"x": (1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                   "y": (1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE)},
            order=3)
        self.scale_transform = iaa.Affine(scale=(1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                                          order=3)
        self.USE_SKEWED_SCALE_RATE = cfg.AUGMENTATIONS.USE_SKEWED_SCALE_RATE
        self.SIMULATE_XRAY_ARTEFACTS_RATE = cfg.AUGMENTATIONS.SIMULATE_XRAY_ARTEFACTS_RATE
        # augmentations

        # self.affine = A.Compose(
        #     [A.Rotate(limit=cfg.AUGMENTATIONS.ROTATION, p=0.4),
        #      A.Affine(  # rotate=(-cfg.AUGMENTATIONS.ROTATION * 2, cfg.AUGMENTATIONS.ROTATION * 2),
        #          scale=(1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
        #          translate_px={"x": cfg.AUGMENTATIONS.TRANSLATION_X,
        #                        "y": cfg.AUGMENTATIONS.TRANSLATION_Y},
        #          p=0.9)], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        self.transform = A.Compose([

            # A.Rotate(limit=cfg.AUGMENTATIONS.ROTATION, p=0.4),
            A.Affine(rotate=(-cfg.AUGMENTATIONS.ROTATION, cfg.AUGMENTATIONS.ROTATION),
                     scale=(1 - cfg.AUGMENTATIONS.SCALE, 1 + cfg.AUGMENTATIONS.SCALE),
                     translate_px={"x": cfg.AUGMENTATIONS.TRANSLATION_X,
                                   "y": cfg.AUGMENTATIONS.TRANSLATION_Y},
                     p=0.5),
            A.Erasing(p=0.75, scale=(cfg.AUGMENTATIONS.CUTOUT_SIZE_MIN, cfg.AUGMENTATIONS.CUTOUT_SIZE_MAX)),

            A.Equalize(p=0.3),

            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.3),
            A.GaussNoise(std_range=(0, 0.1), p=0.2),
            A.AdvancedBlur(sigma_x_limit=(0.2, 1.0), sigma_y_limit=(0.2, 1.0), p=0.3),
            A.HorizontalFlip(p=0.5),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            # A.ElasticTransform(alpha=70, sigma=30,
            #                    p=1,
            #                    keypoint_remapping_method="direct"
            #                    )
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        self.coarse_dropout = iaa.CoarseDropout(0.02, size_percent=0.08)
        self.addative_gaussian_noise = iaa.AdditiveGaussianNoise(scale=(0, cfg.AUGMENTATIONS.GAUSSIAN_NOISE * 255))
        self.coarse_dropout_rate = cfg.AUGMENTATIONS.COARSE_DROPOUT_RATE
        self.addative_gaussian_noise_rate = cfg.AUGMENTATIONS.ADDATIVE_GAUSSIAN_NOISE_RATE

        self.FLIP_INITIAL_COORDINATES = cfg.AUGMENTATIONS.FLIP_INITIAL_COORDINATES

        self.RADIUS = cfg.TRAINLOSSES.MASK_RADIUS

        self.cfg_INT_TO_FLOAT = cfg.DATASET.INT_TO_FLOAT
        self.cfg_NORMALISATION_METHOD = cfg.DATASET.NORMALISATION
        self.cfg_BCE_weight = cfg.TRAINLOSSES.BCE_WEIGHT

        self.store_in_ram = True
        self.ram = defaultdict(dict)

        self.normalise = normalise

        self.resize = iaa.Sequential([
            iaa.PadToAspectRatio(cfg.DATASET.IMG_SIZE[1] / cfg.DATASET.IMG_SIZE[0],
                                 position="uniform" if self.augment else "right-bottom"),
            iaa.Resize({"height": cfg.DATASET.IMG_SIZE[0], "width": cfg.DATASET.IMG_SIZE[1]})
        ])

    def __len__(self):
        return len(self.data)

    def handle_partition(self):
        """
        training | validation | testing
        :return:
        """
        with open(self.partition_file, "r") as partition_file:
            self.data = json.load(partition_file)[self.partition]
        self.data_set = set(self.data)

        for file in os.listdir(self.CACHE_DIR + "/" + self.DATASET_NAME):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"):
                file_name = file.split(".")[0]

                if file_name in self.data_set:
                    self.images.append(f"{self.CACHE_DIR}/{self.DATASET_NAME}/{file}")
                    self.annotations.append(f"{self.CACHE_DIR}/{self.DATASET_NAME}/{file_name}_annotations.txt")
                    self.metas.append(f"{self.CACHE_DIR}/{self.DATASET_NAME}/{file_name}_meta.json")

    def img_int_to_float(self, img):
        # # 0-255 (none), image/255 (standard), max-min/max (0-1), adaptive
        int_to_float = self.cfg_INT_TO_FLOAT
        if self.cfg_INT_TO_FLOAT == "random" and self.augment:
            int_to_float = \
                random.choices(["none", "standard", "minmax", "adaptive"], weights=[0.25, 0.25, 0.25, 0.25])[
                    0]
        elif self.cfg_INT_TO_FLOAT == "random" and not self.augment:
            int_to_float = "standard"

        if int_to_float == "none":
            return img.astype(np.float32)
        elif int_to_float == "standard":
            return img.astype(np.float32) / 255
        elif int_to_float == "minmax":
            img = img.astype(np.float32)
            img -= img.min()
            img /= img.max()
            return img
        elif int_to_float == "adaptive":
            return skimage.exposure.equalize_adapthist(img)
        else:
            raise ValueError(f"Invalid int to float method {self.cfg_INT_TO_FLOAT}")

    # @profile
    def __getitem__(self, idx):

        # load image
        # load label
        # augment
        # generate landmark image
        # return dictionary with image, landmark, & landmark image and filename

        # if self.augment:
        #     imgaug.seed(np.random.randint(0, 10000))
        #     np.random.seed(np.random.randint(0, 10000))
        image_name = self.images[idx]
        annotation_file = self.annotations[idx]
        meta_file = self.metas[idx]

        # load data
        if self.store_in_ram and idx not in self.ram:
            if self.USE_GRAYSCALE:
                img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = io.imread(image_name, as_gray=True)
            self.ram[idx] = img
        elif self.store_in_ram:
            img = self.ram[idx]
        else:
            if self.USE_GRAYSCALE:
                img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.uint8)
        landmarks = np.loadtxt(annotation_file, delimiter=",", max_rows=self.total_landmarks)
        metas = json.load(open(meta_file, "r"))

        # landmarks_all_annotators = np.array(landmarks_all_annotators).reshape(-1, self.total_landmarks, 2)
        # landmarks = landmarks_all_annotators.mean(axis=0)

        # preprocess and augment
        kps = KeypointsOnImage.from_xy_array(landmarks, shape=img.shape)

        # plot keypoints over image for testing
        img, kps = self.resize(image=img, keypoints=kps)
        kps = kps.to_xy_array().reshape(-1, 2)

        if self.augment:
            if random.random() < self.SIMULATE_XRAY_ARTEFACTS_RATE:
                img = simulate_x_ray_artefacts(img)

            transformed_data = self.transform(image=img, keypoints=kps)
            img_aug = transformed_data['image']
            kps_aug = transformed_data['keypoints']

            img, kps = img_aug, kps_aug

            # if kps_aug.shape[0] != self.total_landmarks:
            #     plt.imshow(img_aug, cmap="gray")
            #     plt.scatter(kps_aug[:, 0], kps_aug[:, 1], c='r', s=2)
            #     # plt.title(f"Image {batch['name'][0]}")
            #     plt.show()

        kps = np.flip(kps, axis=-1).astype(np.float32)

        img = self.img_int_to_float(img)
        img = self.normalise(img, method=self.cfg_NORMALISATION_METHOD)
        img = img.astype(np.float32)
        output = {"x": img, "y": kps, "name": f"{image_name.split('/')[-1].split('.')[0].split('_')[0]}"}
        if len(output["x"].shape) == 2:
            output["x"] = np.expand_dims(output["x"], axis=0)

        landmarks_rounded = np.round(kps).astype(int)

        # if self.cfg_BCE_weight > 0:
        #     output["y_img_radial"] = create_radial_mask(landmarks_rounded, img.shape, self.dataset_pixels_per_mm,
        #                                                 radius=self.RADIUS)
        output["y_img_initial"] = create_landmark_image(
            landmarks_rounded,
            img.shape,
            use_gaussian=self.USE_GAUSSIAN,
            sigma=self.SIGMAS)

        if self.cfg_BCE_weight > 0:
            output["y_img_radial"] = (output["y_img_initial"] - output["y_img_initial"].min()) / (
                    output["y_img_initial"].max() - output["y_img_initial"].min())

        output["y_img"] = output["y_img_initial"]
        output["pixel_per_mm"] = np.array([metas["pixels_per_mm"], metas["pixels_per_mm"]], dtype=np.float32)
        output["pixel_size"] = (np.array([metas["pixels_per_mm"], metas["pixels_per_mm"]], dtype=np.float32) *
                                metas["scale_factor"])
        output["scale_factor"] = np.array(metas["scale_factor"], dtype=np.float32)
        return output

    @classmethod
    def get_loaders(cls, cfg, batch_size, num_workers, augment_train=False, partition="training", shuffle=None):
        if shuffle is None:
            shuffle = partition == "training"
        dataset = cls(cfg, partition=partition, augment=augment_train and partition == "training")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, persistent_workers=False, pin_memory=False)

        return dataloader


class LandmarkDataModule(L.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers, augment_train=False):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment_train = augment_train

        self.pin_memory = True

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LandmarkDataset(self.cfg, partition="training", augment=True)
            self.val_dataset = LandmarkDataset(self.cfg, partition="validation", augment=False)

        if stage == "test" or stage is None:
            self.test_dataset = LandmarkDataset(self.cfg, partition="testing", augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


def main():
    plt.rcParams["figure.figsize"] = (9, 9)
    plt.rcParams["image.cmap"] = "gray"
    plt.rcParams["figure.dpi"] = 200

    cfg = config.get_config("configs/local_test_ceph_MICCAI24.yaml")
    cfg.DATASET.IMG_SIZE = (448, 448)
    # cfg = config.get_config("configs/example_config.yaml")
    cfg.AUGMENTATIONS.CUTOUT_SIZE_MAX = 0.1
    cfg.DATASET.USE_GAUSSIAN_GT = True
    cfg.AUGMENTATIONS.ROTATION = 10
    cfg.AUGMENTATIONS.SCALE = 0.15
    cfg.DATASET.GT_SIGMA = 1
    cfg.DATASET.CHANNELS = 3

    cache_data(cfg)

    train_loader = LandmarkDataset.get_loaders(cfg, 8, 0, True, partition="training", shuffle=True)
    # train_loader = LandmarkDataset.get_loaders(cfg, 1, 1, False, partition="validation", shuffle=False)
    # train_loader = LandmarkDataset.get_loaders(cfg, 1, 1, False, partition="testing", shuffle=True)
    start = time.time()

    for b, batch in enumerate(train_loader):
        # print(batch.keys())
        pass
        x = batch["x"]
        # print(batch["pixel_size"], batch["scale_factor"], batch["pixel_per_mm"])
        # print(x.min(), x.max(), x.float().mean(), x.float().std(), x.dtype, batch['name'])
        # if batch["name"][0] == "454":
        # for k in batch.keys():
        #     if k != "name":
        #         print(k, batch[k].shape, batch[k].dtype)
        plot_landmarks_from_img(x, batch["y_img_initial"], batch["y_img_initial"], plot=True)
        # plt.imshow((renormalise(x[0], True)).clamp(0, 255).long().cpu().numpy())
        # plt.scatter(batch["y"][0, :, 1], batch["y"][0, :, 0], c='r', s=2)
        # plt.title(f"Image {batch['name'][0]}")
        # plt.show()

        if b > 8:
            break
    print(time.time() - start)


if __name__ == "__main__":
    # get_mean_std()
    main()
    # mnist_loader(4, 1)
    # check_file()
    # check_partitions()
