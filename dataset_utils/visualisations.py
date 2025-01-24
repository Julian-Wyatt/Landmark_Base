import numpy as np
import torch
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
