"""Preprocessing utilities for image normalization and conversion."""

import numpy as np
from typing import Tuple


def normalize(img: np.ndarray, percentile: Tuple[float, float] = (0, 100)) -> np.ndarray:
    """
    Normalize image to [0, 1] range with optional percentile clipping.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    percentile : Tuple[float, float], optional
        Lower and upper percentiles for clipping, by default (0, 100).

    Returns
    -------
    np.ndarray
        Normalized image.
    """
    img_norm = img.copy().astype(np.float32)

    # Compute percentiles
    p_low = np.percentile(img_norm, percentile[0])
    p_high = np.percentile(img_norm, percentile[1])

    # Clip and normalize
    img_norm = np.clip(img_norm, p_low, p_high)

    if p_high > p_low:
        img_norm = (img_norm - p_low) / (p_high - p_low)
    else:
        img_norm = np.zeros_like(img_norm)

    return img_norm


def denormalize(
    img: np.ndarray,
    original_min: float,
    original_max: float
) -> np.ndarray:
    """
    Denormalize image from [0, 1] to original range.

    Parameters
    ----------
    img : np.ndarray
        Normalized image in [0, 1] range.
    original_min : float
        Original minimum value.
    original_max : float
        Original maximum value.

    Returns
    -------
    np.ndarray
        Denormalized image.
    """
    return img * (original_max - original_min) + original_min


def to_float(img: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 in [0, 1] range.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Float32 image in [0, 1] range.
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    elif img.dtype in [np.float32, np.float64]:
        return img.astype(np.float32)
    else:
        # Normalize to [0, 1] for unknown dtypes
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return ((img - img_min) / (img_max - img_min)).astype(np.float32)
        return img.astype(np.float32)


def to_uint8(img: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Convert image to uint8.

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed to be in [0, 1] if float).
    clip : bool, optional
        Clip values to valid range, by default True.

    Returns
    -------
    np.ndarray
        Uint8 image.
    """
    if clip:
        img = np.clip(img, 0, 1 if img.dtype in [
                      np.float32, np.float64] else 255)

    if img.dtype in [np.float32, np.float64]:
        return (img * 255).astype(np.uint8)
    elif img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    else:
        return img.astype(np.uint8)


def to_uint16(img: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Convert image to uint16.

    Parameters
    ----------
    img : np.ndarray
        Input image (assumed to be in [0, 1] if float).
    clip : bool, optional
        Clip values to valid range, by default True.

    Returns
    -------
    np.ndarray
        Uint16 image.
    """
    if clip:
        img = np.clip(img, 0, 1 if img.dtype in [
                      np.float32, np.float64] else 65535)

    if img.dtype in [np.float32, np.float64]:
        return (img * 65535).astype(np.uint16)
    elif img.dtype == np.uint8:
        return (img * 257).astype(np.uint16)  # 257 = 65535 / 255
    else:
        return img.astype(np.uint16)


def rgb_to_gray(img: np.ndarray, weights: Tuple[float, float, float] = (0.299, 0.587, 0.114)) -> np.ndarray:
    """
    Convert RGB image to grayscale using weighted sum.

    Parameters
    ----------
    img : np.ndarray
        RGB image (H, W, 3).
    weights : Tuple[float, float, float], optional
        Weights for R, G, B channels, by default (0.299, 0.587, 0.114).

    Returns
    -------
    np.ndarray
        Grayscale image (H, W).
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input must be an RGB image with shape (H, W, 3)")

    return np.dot(img[..., :3], weights).astype(img.dtype)
