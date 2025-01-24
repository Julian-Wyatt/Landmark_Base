import random

import numpy as np
import scipy
import torch
from numba import njit, prange


def normalise(x, method="none"):
    # 0-1 (none), -1-1 (mu=0.5, sigma=0.5), dataset mean and std
    if method == "none":
        return x
    elif method == "mu=0.5,sig=0.5":
        return (x - 0.5) / 0.5
    elif method == "dataset":
        mu = 0.4469038915295547
        sigma = 0.27797534502741916
        return (x - mu) / sigma
    else:
        return x


def renormalise(x, img=True, method="none"):
    # 0-1 (none), -1-1 (mu=0.5, sigma=0.5), dataset mean and std
    if method == "mu=0.5,sig=0.5":
        norm = (x * 0.5) + 0.5
    elif method == "dataset":
        mu = 0.4469038915295547
        sigma = 0.27797534502741916
        norm = (x * sigma) + mu
    else:
        norm = x
    if img and norm.mean() < 3:
        return norm * 255
    return norm


def simulate_x_ray_artefacts(image):
    # randomly add noise or multiply by 0-0.1 to a random slice of the image if
    input_dtype = image.dtype
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32)

    # randomly decide height or width
    axis = random.choice([0, 1])

    # decide size of slice (25,50,75 pixels)
    slice_size = random.choice([25, 50, 75, 100, 125])
    # choose start_idx
    start_idx = random.randint(0, image.shape[axis] - slice_size)
    end_idx = start_idx + slice_size
    if axis == 0:  # height
        slice_mask = np.s_[start_idx:end_idx, :]
    else:  # width
        slice_mask = np.s_[:, start_idx:end_idx]
    # Generate noise or scale factor
    if random.choice([True, False]):  # Apply noise
        np.add(image[slice_mask], np.random.normal(0, 15, image[slice_mask].shape), out=image[slice_mask])
    else:  # Apply scaling
        np.multiply(image[slice_mask], np.random.uniform(0.5, 1.5), out=image[slice_mask])

    image.clip(min=0, max=255, out=image)
    return image.astype(input_dtype)


def get_coordinates_from_heatmap(heatmap: torch.tensor, k=1, threshold=0.5):
    # heatmap is in the format [B, C, H, W]
    # coordinates are in the format [B, C, 2]
    # get the value and flattened index of the heatmap over each channel
    b, c, _, _ = heatmap.shape
    max_values, max_indices = torch.topk(heatmap.view(heatmap.shape[0], heatmap.shape[1], -1), dim=2, k=k)

    # add a spatial dimension
    max_values = max_values.unsqueeze(-1)
    max_indices = max_indices.unsqueeze(-1)
    # convert the flattened index to 2D coordinates
    coordinates = torch.cat([max_indices % heatmap.shape[3], max_indices // heatmap.shape[3]], dim=-1)
    if torch.isclose(torch.sum(max_values, dtype=max_values.dtype),
                     torch.zeros((1,), dtype=max_values.dtype, device=max_values.device)):
        coords = torch.zeros((b, c, 2), device=coordinates.device) - 1
        return coords.float()

    # Softmax the values across the k channel
    # max_values = torch.nn.functional.softmax(max_values, dim=2)
    max_values = max_values / torch.sum(max_values, dim=2, keepdim=True)
    # average the coordinates of the top k values by their value
    coordinates = torch.sum(coordinates * max_values, dim=2)

    return coordinates.float()


# @njit(parallel=False, fastmath=True)
# TODO Allow above to work with non-rounded landmarks - ie via antialiasing
# TODO: Fix the parallelisation - breaks with Gaussian filter
def create_landmark_image(landmarks, img_size, use_gaussian=False, sigma=None):
    """Convert coordinates to image with landmarks as neighbourhoods around the coordinates"""
    c, d = landmarks.shape
    h, w = img_size
    landmark_img = np.zeros((c, h, w), dtype=np.float32)
    if sigma is None:
        sigma = np.ones(c) * 1.0

    for k in prange(c):
        # round the landmark to the nearest integer
        x, y = landmarks[k, 1], landmarks[k, 0]
        if x < 0 or y < 0 or y >= h or x >= w:
            continue
        landmark_img[k, y, x] = 1.0
        if use_gaussian and sigma[k] > 0:
            scipy.ndimage.gaussian_filter(landmark_img[k], sigma=sigma[k], output=landmark_img[k])
            # landmark_img[k] = skimage.filters.gaussian(landmark_img[k], sigma=sigma[k])

    return landmark_img


def create_radial_mask(landmarks, img_size, pixel_size, device="cpu", radius=4, min_radius=0):
    """
    Create a radial mask around the landmarks
    """
    h, w = img_size
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing='ij')
    mask = torch.zeros((landmarks.shape[0], h, w), device=device)
    for i, landmark in enumerate(landmarks):
        if landmark[0] < 0 or landmark[1] < 0 or landmark[0] >= h or landmark[1] >= w:
            continue
        x, y = landmark[1], landmark[0]

        distances = torch.sqrt(((yy - y) * pixel_size[0]) ** 2 + ((xx - x) * pixel_size[1]) ** 2)

        # mask[i] = torch.where((min_radius <= distances) & (distances <= radius))
        mask[i] = torch.where((min_radius <= distances) & (distances <= radius), 1, 0)

    # return torch.nonzero(mask, as_tuple=True)

    return mask


def create_radial_mask_batch(landmarks, img_size, pixel_size, device="cpu", radius=4, min_radius=0, do_normalise=True):
    """
    Create a radial mask around the landmarks
    """
    h, w = img_size
    mask = torch.zeros((landmarks.shape[0], landmarks.shape[1], h, w), device=device)
    for i in range(landmarks.shape[0]):
        mask[i] = create_radial_mask(landmarks[i], img_size, pixel_size[i], device, radius, min_radius, do_normalise)
    return mask
