import torch
import numpy as np

from dataset_utils.preprocessing_utils import get_coordinates_from_heatmap


def euclidean_distance(x, y, use_torch=True):
    if use_torch:
        return torch.linalg.norm(x - y, dim=-1)
    else:
        return np.linalg.norm(x - y, axis=-1)


def manhattan_distance(x, y, use_torch=True):
    if use_torch:
        return torch.sum(torch.abs(x - y), dim=-1)
    else:
        return np.sum(np.abs(x - y), axis=-1)


def success_detection_rates(radial_errors, thresholds=None):
    if thresholds is None:
        thresholds = [5, 10, 20, 40]
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * np.sum(radial_errors < threshold) / len(radial_errors)
        successful_detection_rates.append(sdr)
    return successful_detection_rates


def calculate_ere(heatmap, predicted_point_scaled, pixel_size=1, significant_pixel_cutoff=0.05, epoch=0):
    # TODO: FIX THIS FUNCTION
    # https://github.com/jfm15/ContourHuggingHeatmaps/blob/main/evaluate.py
    b, k, h, w = heatmap.shape
    normalized_heatmap = heatmap / torch.max(heatmap)
    normalized_heatmap = torch.where(normalized_heatmap > significant_pixel_cutoff, normalized_heatmap, 0)
    normalized_heatmap /= torch.sum(normalized_heatmap, dim=(2, 3), keepdim=True)
    # get all non-zero item indices of shape [N,4] for N non-zero items and input of size [B,K,H,W]
    indices = torch.argwhere(normalized_heatmap)
    # output shape is [B,K]
    output = torch.zeros(b, k, device=predicted_point_scaled.device, dtype=normalized_heatmap.dtype)
    indices_b, indices_k = indices[:, 0], indices[:, 1]
    scaled_idx = indices[:, 2:] * pixel_size
    dist = euclidean_distance(predicted_point_scaled[indices_b, indices_k], scaled_idx).float()
    output[indices_b, indices_k] = dist * normalized_heatmap[indices_b, indices_k, indices[:, 2], indices[:, 3]]
    return output


@torch.no_grad()
def evaluate_landmark_detection(heatmaps, real_landmarks, pixel_sizes: torch.Tensor = None, epoch=1,
                                top_k=1):
    # return log with manhatten distance, euclidean distance, and expected error
    if pixel_sizes is None:
        pixel_sizes = torch.ones(heatmaps.shape[0], 2, device=heatmaps.device)
    heatmaps = heatmaps.float()
    real_landmarks = real_landmarks.float()
    log = {}
    pixel_sizes = pixel_sizes.unsqueeze(1)
    # x,y landmarks
    coordinates = get_coordinates_from_heatmap(heatmaps, k=top_k).flip(-1)
    coordinates *= pixel_sizes
    real_landmarks = real_landmarks.float() * pixel_sizes

    # shape [B, K] for l1, l2 and ere
    log["l1"] = manhattan_distance(coordinates, real_landmarks).cpu().numpy()
    log["l2"] = euclidean_distance(coordinates, real_landmarks).cpu().numpy()
    log["ere"] = calculate_ere(heatmaps, coordinates, epoch=epoch).cpu().numpy()

    return log


@torch.no_grad()
def evaluate_landmark_detection_no_heatmap(predicted_landmarks, real_landmarks, pixel_sizes: torch.Tensor = None):
    # return log with manhatten distance, euclidean distance, and expected error
    if pixel_sizes is None:
        pixel_sizes = torch.ones(predicted_landmarks.shape[0], 2, device=predicted_landmarks.device)
    predicted_landmarks = predicted_landmarks.float()
    real_landmarks = real_landmarks.float()
    log = {}
    pixel_sizes = pixel_sizes.unsqueeze(1)
    # x,y landmarks
    coordinates = predicted_landmarks
    coordinates *= pixel_sizes
    real_landmarks = real_landmarks.float() * pixel_sizes

    # shape [B, K] for l1, l2 and ere
    log["l1"] = manhattan_distance(coordinates, real_landmarks).cpu().numpy()
    log["l2"] = euclidean_distance(coordinates, real_landmarks).cpu().numpy()
    return log


def main_test():
    a = torch.tensor([[2, 1], [3, 1], [0, 0], [0, 1], [4, 4]]).float()
    b = torch.tensor([[4, 5], [3, 4], [7, 7], [8, 8], [4, 4]]).float()
    print(euclidean_distance(a.unsqueeze(0), b.unsqueeze(0)))
    print("TEST 1")
    real = torch.Tensor([[[281., 36.],
                          [280., 90.],
                          [316., 148.],
                          [378., 182.],
                          [230., 190.]],

                         [[313., 43.],
                          [311., 102.],
                          [362., 185.],
                          [398., 209.],
                          [261., 221.]]])
    pred = torch.Tensor([[[284., 28.],
                          [282., 103.],
                          [315., 148.],
                          [376., 180.],
                          [235., 183.]],
                         [[313., 56.],
                          [309., 111.],
                          [357., 184.],
                          [409., 227.],
                          [268., 216.]]])
    print(real.shape, pred.shape)
    print(euclidean_distance(real, pred))


if __name__ == "__main__":
    main_test()
    # main_test("angle")
