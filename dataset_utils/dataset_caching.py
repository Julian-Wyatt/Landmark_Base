import glob
import json
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imgaug import KeypointsOnImage
from tqdm import tqdm

from core import config
from core.config import Config
import imgaug.augmenters as iaa


def cache_data(cfg: Config):
    resize_dir = os.path.join(cfg.DATASET.CACHE_DIR, cfg.DATASET.NAME,
                              f"{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}")
    annotations_df = pd.read_csv(f"{cfg.DATASET.ROOT_DIR}/{cfg.DATASET.ANNOTATIONS_FILE}")

    annotations_df["image file"] = annotations_df["image file"].str.split('.').str[0]

    # loop over all image files
    # for each image file, load the image and the landmarks
    # resize the image
    # save the image and the landmarks w/ scale factor

    if not os.path.exists(resize_dir):
        os.makedirs(resize_dir)

    files = []
    for i in cfg.DATASET.IMG_DIRS:
        files.extend(glob.glob(f"{cfg.DATASET.ROOT_DIR}{'/' if cfg.DATASET.ROOT_DIR[-1] != '/' else ''}{i}/*"))
    for file in tqdm(sorted(files)):
        image_name = file.split("/")[-1].split(".")[0]
        # if image_name not in data_set:
        #     continue

        cached_image_name = f"{resize_dir}/{image_name}.png"
        cached_meta_name = f"{resize_dir}/{image_name}_meta.json"
        annotations_name = f"{resize_dir}/{image_name}_annotations.txt"

        # if the image has not been resized
        if not os.path.exists(cached_image_name):
            # if True:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            annotation_original_image_shape = image.shape
            # preprocess from dataset_utils

            # resize the image
            resize = iaa.Resize({"height": cfg.DATASET.IMG_SIZE[0], "width": "keep-aspect-ratio"})
            operations = {"pre-resize shape": image.shape}

            # if type(cfg.DATASET.dataset_pixels_per_mm) is list:
            #     pixels_per_mm = cfg.DATASET.dataset_pixels_per_mm[0]
            # else:
            #     pixels_per_mm = cfg.DATASET.dataset_pixels_per_mm

            NO_LABEL = False
            image_data = annotations_df.loc[annotations_df["image file"] == f"{image_name}"]
            if image_data.empty:
                # raise FileNotFoundError(
                #     f"Annotation file not found - image {self.root_dir}{'/' if self.root_dir[-1] != '/' else ''}{annotation_dir}/{image_name}.txt")
                landmarks = np.zeros((cfg.DATASET.NUMBER_KEY_POINTS, 2))
                pixels_per_mm = 1
                NO_LABEL = True
            else:
                if image_data.keys()[2] == "label":
                    landmarks = image_data.iloc[0, 3:].values.astype('float').reshape(-1, 2)
                    pixels_per_mm = annotations_df.loc[annotations_df["image file"] == f"{image_name}"].iloc[0, 2]
                else:
                    landmarks = image_data.iloc[0, 2:].values.astype('float').reshape(-1, 2)
                    pixels_per_mm = annotations_df.loc[annotations_df["image file"] == f"{image_name}"].iloc[0, 1]

            operations["image_name"] = image_name
            # Key

            if NO_LABEL:
                landmarks = np.zeros((cfg.DATASET.NUMBER_KEY_POINTS, 2))
            # if random.random() < 0.5:
            #     plt.imshow(image)
            #     plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=2)
            #     # plt.title(f"Image {batch['name'][0]}")
            #     plt.show()
            kps = KeypointsOnImage.from_xy_array(landmarks, shape=image.shape)

            img, kps = resize(image=image, keypoints=kps)
            cv2.imwrite(cached_image_name, img)
            landmarks = kps.to_xy_array()
            landmarks.reshape(-1, 2)

            np.savetxt(
                annotations_name, landmarks, fmt="%.14g", delimiter=","
            )
            original_image_height, original_image_width = operations['pre-resize shape']
            original_aspect_ratio = original_image_width / original_image_height
            downsampled_aspect_ratio = cfg.DATASET.IMG_SIZE[1] / cfg.DATASET.IMG_SIZE[0]
            if original_aspect_ratio > downsampled_aspect_ratio:
                scale_factor = original_image_width / cfg.DATASET.IMG_SIZE[1]
            else:
                scale_factor = original_image_height / cfg.DATASET.IMG_SIZE[0]

            with open(cached_meta_name, "w") as meta_file:
                json.dump({"scale_factor": scale_factor, "filename": image_name,
                           "pixels_per_mm": pixels_per_mm,
                           "pre-resize shape": operations['pre-resize shape'], "no_label_exists": NO_LABEL},
                          meta_file)
            # if random.random() < 0.05:
            #     plt.imshow(img)
            #     plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=2)
            #     # plt.title(f"Image {batch['name'][0]}")
            #     plt.show()


# Dataset standardisation
def CephAdoAdu(dir=""):
    # pixels per mm set to 0.1 from their github repo
    pixels_per_mm = 0.1

    # print set difference of adult image and txt folders and under_age image and txt folders
    adult_images = set([i.split(".")[0] for i in os.listdir(f"{dir}/adult/dataset")])
    adult_labels = set([i.split(".")[0] for i in os.listdir(f"{dir}/adult/txt")])
    under_age_images = set([i.split(".")[0] for i in os.listdir(f"{dir}/under_age/dataset")])
    under_age_labels = set([i.split(".")[0] for i in os.listdir(f"{dir}/under_age/txt")])
    print(adult_labels.symmetric_difference(adult_images))
    print(under_age_labels.symmetric_difference(under_age_images))

    total_labels = pd.DataFrame(
        columns=["image file", "spacing(mm)", "label"] + [f"p{i}{dim}" for i in range(1, 11) for dim in ["x", "y"]])
    for subset_dir in ["adult", "under_age"]:
        for image in os.listdir(f"{dir}/{subset_dir}/dataset"):
            if image.endswith(".jpg"):
                image_name = image.split(".")[0]
                image_data = {
                    "image file": f"{image_name}.jpg",
                    "spacing(mm)": pixels_per_mm,
                    "label": subset_dir
                }

                # read label data from txt
                # if not os.path.exists(f"{dir}/{subset_dir}/txt/{image_name}.txt"):
                #     # print(f"Label file not found for {image_name}")
                #     continue
                try:
                    with open(f"{dir}/{subset_dir}/txt/{image_name}.txt") as f:
                        landmarks = json.load(f)
                    for i in range(1, 11):
                        image_data[f"p{i}x"] = landmarks[i - 1]['data'][0]["x"]
                        image_data[f"p{i}y"] = landmarks[i - 1]['data'][0]["y"]
                except FileNotFoundError:
                    for i in range(1, 11):
                        image_data[f"p{i}x"] = -1
                        image_data[f"p{i}y"] = -1

                # for i in range(1, 11):
                #     image_data[f"p{i}x"] = landmarks[i - 1]['data'][0]["x"]
                #     image_data[f"p{i}y"] = landmarks[i - 1]['data'][0]["y"]

                # Append as a single-row DataFrame
                total_labels = pd.concat([total_labels, pd.DataFrame([image_data])], ignore_index=True)

    #     save to csv
    total_labels.to_csv(f"{dir}/labels.csv", index=False)


def ISBI2015(dir=""):
    # TODO: Convert txt annotations to csv
    pass


def MICCAI2024(dir=""):
    """

    001 to 077 inclusive need to swap 789 to 798
    535 swap 9 and 16, then swap 8 and 9
    """
    landmarks_df = pd.read_csv(f"{dir}/Training Set/labels.csv")
    for i in range(1, 78):
        # files are zero padded to three digits
        landmarks_df.loc[landmarks_df["image file"] == f"{i:03d}.bmp", ["p7x", "p7y", "p8x", "p8y", "p9x", "p9y"]] = \
            landmarks_df.loc[
                landmarks_df["image file"] == f"{i:03d}.bmp", ["p7x", "p7y", "p9x", "p9y", "p8x", "p8y"]].values

    landmarks_df.loc[landmarks_df["image file"] == "535.bmp", ["p9x", "p9y", "p16x", "p16y"]] = \
        landmarks_df.loc[landmarks_df["image file"] == "535.bmp", ["p16x", "p16y", "p9x", "p9y"]].values
    landmarks_df.loc[landmarks_df["image file"] == "535.bmp", ["p8x", "p8y", "p9x", "p9y"]] = \
        landmarks_df.loc[landmarks_df["image file"] == "535.bmp", ["p9x", "p9y", "p8x", "p8y"]].values

    #     landmarks_df.loc[landmarks_df["image file"] == f"{i:03d}.bmp", ["p6x", "p6y", "p7x", "p7y", "p8x", "p8y"]] = \
    #         landmarks_df.loc[
    #             landmarks_df["image file"] == f"{i:03d}.bmp", ["p6x", "p6y", "p8x", "p8y", "p7x", "p7y"]].values
    #
    # landmarks_df.loc[landmarks_df["image file"] == "535.bmp", ["p15x", "p15y", "p8x", "p8y"]] = \
    #     landmarks_df.loc[landmarks_df["image file"] == "535.bmp", ["p8x", "p8y", "p15x", "p15y"]].values

    landmarks_df.to_csv(f"{dir}/Training Set/labels_fix_v2.csv", index=False)


def visualise_cached_data(cfg: Config, show_landmark_numbers=False, show_landmark_points=True):
    # plot landmark numbers over each image in dataset
    # save images in a temp vis folder
    vis_dir = f"{cfg.DATASET.CACHE_DIR}/{cfg.DATASET.NAME}/{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}/vis"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    for i in tqdm(glob.glob(
            f"{cfg.DATASET.CACHE_DIR}/{cfg.DATASET.NAME}/{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}/*_meta.json")):
        with open(i, "r") as f:
            meta = json.load(f)
        image = cv2.imread(
            f"{cfg.DATASET.CACHE_DIR}/{cfg.DATASET.NAME}/{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}/{meta['filename']}.png")
        landmarks = np.loadtxt(
            f"{cfg.DATASET.CACHE_DIR}/{cfg.DATASET.NAME}/{cfg.DATASET.IMG_SIZE[0]}x{cfg.DATASET.IMG_SIZE[1]}/{meta['filename']}_annotations.txt",
            delimiter=",")

        if show_landmark_numbers:
            for j in range(0, landmarks.shape[0]):
                cv2.putText(image, str(j + 1), (int(landmarks[j, 0]), int(landmarks[j, 1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
        if show_landmark_points:
            for j in range(0, landmarks.shape[0]):
                cv2.circle(image, (int(landmarks[j, 0]), int(landmarks[j, 1])), 2, (255, 0, 0), -1)
        # plt.imshow(image)
        # plt.show()
        # break
        cv2.imwrite(f"{vis_dir}/{meta['filename']}.png", image)


def visualiseLandmarksOverDataset(image_dir, landmark_file, show_landmark_numbers=False, show_landmark_points=True):
    # plot landmark numbers over each image in dataset
    # save images in a temp vis folder

    if not os.path.exists(f"vis/{image_dir.split('/')[-3]}"):
        os.makedirs(f"vis/{image_dir.split('/')[-3]}")

    landmarks_df = pd.read_csv(landmark_file)
    for i in tqdm(glob.glob(image_dir + "/*")):
        if i.endswith(".png") or i.endswith(".jpg") or i.endswith(".bmp"):
            image = cv2.imread(i)
            current_landmark = landmarks_df.loc[landmarks_df["image file"] == f"{i.split('/')[-1]}"]

            current_landmark = current_landmark.iloc[0, 2:].values.astype('int').reshape(-1, 2)
            if show_landmark_numbers:
                for j in range(0, current_landmark.shape[0]):
                    cv2.putText(image, str(j + 1), (current_landmark[j, 0], current_landmark[j, 1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)
            if show_landmark_points:
                for j in range(0, current_landmark.shape[0]):
                    cv2.circle(image, (current_landmark[j, 0], current_landmark[j, 1]), 8, (0, 0, 255), -1)

            cv2.imwrite(f"vis/{image_dir.split('/')[-3]}/{i.split('/')[-1]}", image)


def main(config_file=""):
    cfg = config.get_config(config_file)
    cache_data(cfg)


if __name__ == "__main__":
    visualise_cached_data(config.get_config(
        "/Users/julatt/Documents/DPhil/Year 2/landmark_detection_base/configs/local_test_ceph_MICCAIAdoAdu24.yaml"))

    # main("/Users/julatt/Documents/DPhil/Year 2/landmark_detection_base/configs/local_test_ceph_MICCAI24.yaml")
    # CephAdoAdu("/Users/julatt/Documents/DPhil/Year 1/DiffLand/datasets/CephAdoAdu Dataset")
    # main("/Users/julatt/Documents/DPhil/Year 2/landmark_detection_base/configs/local_test_ceph_MICCAIAdoAdu24.yaml")
    # MICCAI2024(
    #     "/Users/julatt/Documents/DPhil/Year 1/DiffLand/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge")
    # visualiseLandmarksOverDataset(
    #     "/Users/julatt/Documents/DPhil/Year 1/DiffLand/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images",
    #     "/Users/julatt/Documents/DPhil/Year 1/DiffLand/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/labels_fix_v2.csv")
    # visualiseLandmarksOverDataset(
    #     "/Users/julatt/Documents/DPhil/Year 1/DiffLand/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/images",
    #     "/Users/julatt/Documents/DPhil/Year 1/DiffLand/datasets/datasets-in-use/xray-cephalometric-land/2024-MICCAI-Challenge/Training Set/labels_fix_v2.csv")
