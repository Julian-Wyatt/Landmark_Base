import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from dataset_utils.preprocessing_utils import get_coordinates_from_heatmap, renormalise


def plot_heatmaps(heatmaps, ground_truths):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    s = 0
    # print(img.shape, heatmaps.shape, ground_truths.shape)
    # # torch.Size([1, 1, 800, 640]) torch.Size([1, 19, 800, 640]) torch.Size([1, 19, 2])
    # print(img.device, heatmaps.device, ground_truths.device)
    # heatmaps = renormalise(heatmaps, False)
    # Display heatmaps
    normalized_heatmaps = heatmaps / torch.amax(heatmaps, dim=(2, 3), keepdim=True)

    squashed_heatmaps = torch.amax(normalized_heatmaps, dim=1)

    # Display predicted points
    predicted_landmark_positions = get_coordinates_from_heatmap(normalized_heatmaps).cpu().numpy()
    # print(predicted_landmark_positions.shape)
    # print(predicted_landmark_positions)

    # Display ground truth points
    ground_truth_landmark_position = ground_truths[s].cpu().numpy()
    ax.scatter(
        ground_truth_landmark_position[:, 1],
        ground_truth_landmark_position[:, 0],
        color="green",
        s=2,
        alpha=0.2,
    )

    ax.imshow(squashed_heatmaps[s].cpu().numpy(), cmap="gnuplot2", alpha=0.5)
    ax.set_axis_off()
    return fig


def plot_heatmaps_and_landmarks_over_img(img, heatmaps, ground_truths, return_as_array=False,
                                         normalisation_method="none"):
    "https://github.com/jfm15/ContourHuggingHeatmaps/blob/main/evaluate.py"
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    s = 0
    # print(img.shape, heatmaps.shape, ground_truths.shape)
    # # torch.Size([1, 1, 800, 640]) torch.Size([1, 19, 800, 640]) torch.Size([1, 19, 2])
    # print(img.device, heatmaps.device, ground_truths.device)
    heatmaps = heatmaps * 255
    # Display image
    image = renormalise(img[s, 0], False, method=normalisation_method)  # [1, 1, 800, 640] -> [800, 640]
    ax.imshow(image.cpu().numpy(), cmap="gray")
    # Display heatmaps
    heatmaps_thresh = torch.where(heatmaps > 0.05, heatmaps, 0)
    normalized_heatmaps = heatmaps_thresh / torch.amax(heatmaps_thresh, dim=(2, 3), keepdim=True)

    squashed_heatmaps = torch.amax(normalized_heatmaps, dim=1)
    # squashed_heatmaps = torch.where(squashed_heatmaps > 0.05, squashed_heatmaps, 0)

    squashed_heatmaps_np = squashed_heatmaps[s].cpu().numpy()
    heatmap_min, heatmap_max = np.min(squashed_heatmaps_np), np.max(squashed_heatmaps_np)
    norm = mcolors.Normalize(vmin=heatmap_min, vmax=heatmap_max)
    heatmap_colored = plt.cm.gnuplot2(norm(squashed_heatmaps_np))

    # Display predicted points
    predicted_landmark_positions = get_coordinates_from_heatmap(heatmaps).cpu().numpy()
    # print(predicted_landmark_positions.shape)
    # print(predicted_landmark_positions)

    # Display ground truth points
    ground_truth_landmark_position = ground_truths[s].cpu().numpy()
    ax.scatter(
        ground_truth_landmark_position[:, 1],
        ground_truth_landmark_position[:, 0],
        color="green",
        s=2,
        alpha=0.7,
    )

    ax.scatter(
        predicted_landmark_positions[s, :, 0],
        predicted_landmark_positions[s, :, 1],
        color="red",
        alpha=0.9,
        s=2,
    )

    ax.imshow(heatmap_colored, alpha=0.6)
    ax.set_axis_off()
    if return_as_array:
        ax.margins(x=0)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # convert mplt image to numpy array
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")
        plt.clf()
        return data
    return fig


def plot_landmarks_from_img(img: torch.tensor, landmarks: torch.tensor, true_landmark=None, plot=False):
    """
    Plot landmarks and ground truth landmarks onto query image given image coordinates
    """
    landmarks = get_coordinates_from_heatmap(landmarks)
    if true_landmark is not None:
        true_landmark = get_coordinates_from_heatmap(true_landmark)
    return plot_landmarks(img, landmarks, plot=plot, true_landmark=true_landmark)


def plot_landmarks(img: torch.tensor, landmarks: torch.tensor, true_landmark=None, plot=False):
    """
    Plot landmarks and ground truth landmarks onto query image
    """
    import cv2
    # if img.min() < 0:
    # img = renormalise(img).to(torch.uint8)
    img = img.to("cpu")
    landmarks = landmarks.to("cpu")
    if true_landmark is not None:
        true_landmark = true_landmark.to("cpu")

    # img should be B x C x H x W

    if len(img.shape) == 3:
        img = img.reshape(img.shape[0], 1, img.shape[1], img.shape[2])
    if img.shape[-1] in [1, 3]:
        img = rearrange(img, 'b h w c -> b c h w')
    img = img.permute(0, 2, 3, 1).clip(0, 255).numpy().astype(np.uint8)
    output_img = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
    for i in range(img.shape[0]):
        if img.shape[3] == 1:
            img_color = cv2.cvtColor(img[i], cv2.COLOR_GRAY2BGR)
        else:
            img_color = img[i].copy()
        if true_landmark is not None:
            for landmark in true_landmark[i]:
                if landmark[0] < 0 or landmark[1] < 0 or landmark[1] >= img.shape[1] or landmark[0] >= img.shape[2] or \
                        torch.isnan(landmark[0]) or torch.isnan(landmark[1]):
                    continue
                cv2.circle(img_color, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

        for landmark in landmarks[i]:
            if landmark[0] < 0 or landmark[1] < 0 or landmark[1] >= img.shape[1] or landmark[0] >= img.shape[2] or \
                    torch.isnan(landmark[0]) or torch.isnan(landmark[1]):
                continue
            cv2.circle(img_color, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), -1)

        if plot:
            plt.imshow(img_color.astype(np.uint8))
            plt.show()

        output_img[i] = img_color
    return torch.tensor(output_img).permute(0, 3, 1, 2)
