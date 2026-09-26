from __future__ import annotations

import cv2
import numpy as np

from . import config


def _components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = np.zeros_like(mask)
    kept = 0
    area = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            output[labels == index] = 255
            kept += 1
            area += int(stats[index, cv2.CC_STAT_AREA])
    return output, kept, area


def extract_features(image, hsv=None, lab=None, fruit_mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if hsv is None else hsv
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if lab is None else lab
    if fruit_mask is None:
        raise ValueError("fruit_mask is required")

    inside = fruit_mask > 0
    total = max(1, int(inside.sum()))
    h, saturation, value = cv2.split(hsv)
    lightness, lab_a, lab_b = cv2.split(lab)
    dark = inside & (value < 72) & (saturation > 24)
    green = inside & ~dark & (h >= config.GREEN_H_MIN) & (h <= config.GREEN_H_MAX) & (saturation >= config.MIN_COLOR_SATURATION) & (lab_a <= config.GREEN_LAB_A_MAX)
    yellow = inside & ~dark & (h >= config.YELLOW_H_MIN) & (h <= config.YELLOW_H_MAX) & (saturation >= config.MIN_COLOR_SATURATION) & (value >= config.MIN_COLOR_VALUE) & (lab_b >= config.YELLOW_LAB_B_MIN)
    brown = inside & ~dark & (h >= config.BROWN_H_MIN) & (h <= config.BROWN_H_MAX) & (saturation >= 35) & (value < 205) & (lab_a >= config.BROWN_LAB_A_MIN) & (lab_b >= config.BROWN_LAB_B_MIN)
    brown &= ~green & ~yellow

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_contrast = cv2.GaussianBlur(gray, (0, 0), 11).astype(np.int16) - gray.astype(np.int16)
    chroma = np.abs(lab_a.astype(np.int16) - 128) + np.abs(lab_b.astype(np.int16) - 128)
    spots = ((brown | dark) & (local_contrast >= config.LOCAL_SPOT_CONTRAST) & ((chroma > 24) | brown)).astype(np.uint8) * 255
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    spots, spot_count, spot_pixels = _components(spots, max(config.MIN_SPOT_AREA, int(total * config.SPOT_AREA_RATIO)))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient[~inside] = 0
    gradient_values = gradient[inside]
    texture = float(np.sqrt(np.mean(gradient_values ** 2)) / 255 * 10) if gradient_values.size else 0
    local_variance = cv2.GaussianBlur(gray.astype(np.float32) ** 2, (0, 0), 3) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3) ** 2
    quantized = (gray // 32).astype(np.int32)
    horizontal_pairs = inside[:, 1:] & inside[:, :-1]
    differences = np.abs(quantized[:, 1:] - quantized[:, :-1])[horizontal_pairs]

    percentage = lambda mask: float(mask.sum()) * 100 / total
    color_percentages = [percentage(mask) for mask in (green, yellow, brown, dark)]
    color_masks = {"green": green, "yellow": yellow, "brown": brown, "dark": dark}
    feature_values = {
        "green": color_percentages[0],
        "yellow": color_percentages[1],
        "brown": color_percentages[2],
        "dark": color_percentages[3],
        "brown_area": percentage(brown | dark),
        "color_percentage_std": float(np.std(color_percentages)),
        "color_percentage_variance": float(np.var(color_percentages)),
        "brown_spot_percentage": percentage(brown | dark),
        "detected_spot_percentage": spot_pixels * 100 / total,
        "spot_count": spot_count,
        "spot_density": spot_count / max(1, total / 10000),
        "largest_spot_area": 0,
        "average_spot_area": 0,
    }

    # Histograms summarize within-banana HSV distributions, not background pixels.
    for name, channel, limits, bins in (
        ("hue", h, (0, 180), 8),
        ("saturation", saturation, (0, 256), 4),
        ("brightness", value, (0, 256), 4),
    ):
        histogram = cv2.calcHist([channel], [0], fruit_mask, [bins], list(limits)).flatten()
        histogram = histogram * (100.0 / total)
        for index, bin_percentage in enumerate(histogram):
            feature_values[f"{name}_bin_{index}"] = float(bin_percentage)

    # Split the mask bounding box into four regions to measure regional variation.
    ys, xs = np.where(inside)
    if xs.size:
        midpoint_x = (int(xs.min()) + int(xs.max()) + 1) // 2
        midpoint_y = (int(ys.min()) + int(ys.max()) + 1) // 2
    else:
        midpoint_x, midpoint_y = inside.shape[1] // 2, inside.shape[0] // 2
    regions = (
        (slice(None, midpoint_y), slice(None, midpoint_x)),
        (slice(None, midpoint_y), slice(midpoint_x, None)),
        (slice(midpoint_y, None), slice(None, midpoint_x)),
        (slice(midpoint_y, None), slice(midpoint_x, None)),
    )
    region_color_values = {name: [] for name in color_masks}
    spot_region_values = []
    for region_index, region in enumerate(regions):
        region_mask = inside[region]
        region_area = max(1, int(region_mask.sum()))
        for color_name, mask in color_masks.items():
            region_color_values[color_name].append(float(mask[region].sum()) * 100 / region_area)
        spot_percentage = float((spots[region] > 0).sum()) * 100 / region_area
        spot_region_values.append(spot_percentage)
        feature_values[f"spot_region_{region_index}"] = spot_percentage
    for color_name, values in region_color_values.items():
        feature_values[f"regional_{color_name}_std"] = float(np.std(values))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(spots, 8)
    spot_areas = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)]
    feature_values["largest_spot_area"] = max(spot_areas, default=0)
    feature_values["average_spot_area"] = float(np.mean(spot_areas)) if spot_areas else 0
    bgr_means = image[inside].mean(axis=0) if inside.any() else np.zeros(3)
    for name, channel in (("H", h), ("S", saturation), ("V", value)):
        feature_values[f"mean_{name}"] = float(channel[inside].mean()) if inside.any() else 0
        feature_values[f"std_{name}"] = float(channel[inside].std()) if inside.any() else 0
    feature_values.update({
        "mean_R": float(bgr_means[2]), "mean_G": float(bgr_means[1]), "mean_B": float(bgr_means[0]),
        "mean_L": float(lightness[inside].mean()) if inside.any() else 0,
        "mean_a": float(lab_a[inside].mean()) if inside.any() else 0,
        "mean_b": float(lab_b[inside].mean()) if inside.any() else 0,
        "texture_score": texture,
        "gradient_energy": float(np.mean(gradient_values ** 2)) if gradient_values.size else 0,
        "local_variance": float(local_variance[inside].mean()) if inside.any() else 0,
        "glcm_contrast": float(np.mean(differences.astype(float) ** 2)) if differences.size else 0,
        "glcm_homogeneity": float(np.mean(1 / (1 + differences))) if differences.size else 0,
        "glcm_energy": float(np.mean((1 / (1 + differences)) ** 2)) if differences.size else 0,
        "glcm_correlation": float(np.corrcoef(quantized[:, :-1][horizontal_pairs], quantized[:, 1:][horizontal_pairs])[0, 1]) if differences.size > 1 and np.std(quantized[:, :-1][horizontal_pairs]) > 0 and np.std(quantized[:, 1:][horizontal_pairs]) > 0 else 0,
        "fruit_area": total,
        "fruit_coverage": total * 100 / inside.size,
        "aspect_ratio": (int(xs.max()) - int(xs.min()) + 1) / max(1, int(ys.max()) - int(ys.min()) + 1) if xs.size else 0,
    })

    color_visual = np.zeros((*gray.shape, 3), np.uint8)
    color_visual[inside] = (100, 100, 100)
    color_visual[green] = (70, 160, 78)
    color_visual[yellow] = (35, 205, 248)
    color_visual[brown] = (44, 82, 132)
    color_visual[dark] = (40, 40, 40)
    return {
        **{key: round(value, 3) for key, value in feature_values.items()},
        "color_mask": color_visual,
        "green_mask": green.astype(np.uint8) * 255,
        "yellow_mask": yellow.astype(np.uint8) * 255,
        "brown_mask": brown.astype(np.uint8) * 255,
        "brown_spot_mask": spots,
        "gradient": cv2.convertScaleAbs(gradient),
    }


def analyze_colors(hsv, fruit_mask, image=None, lab=None):
    source = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) if image is None else image
    return extract_features(source, hsv, lab, fruit_mask)
