"""Image quality metrics."""

import numpy as np
from typing import Tuple
import cv2


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.

    Returns
    -------
    float
        MSE value (lower is better).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    return np.mean((img1 - img2) ** 2)


def psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image (reference).
    img2 : np.ndarray
        Second image (test).
    max_value : float, optional
        Maximum possible pixel value, by default 1.0.

    Returns
    -------
    float
        PSNR value in dB (higher is better).
    """
    mse_value = mse(img1, img2)

    if mse_value == 0:
        return float('inf')

    return 20 * np.log10(max_value / np.sqrt(mse_value))


def ssim(img1: np.ndarray, img2: np.ndarray, max_value: float = 1.0) -> float:
    """
    Compute Structural Similarity Index between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image (reference).
    img2 : np.ndarray
        Second image (test).
    max_value : float, optional
        Maximum possible pixel value, by default 1.0.

    Returns
    -------
    float
        SSIM value in [0, 1] (higher is better).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    # Convert to appropriate format for cv2.SSIM
    if img1.dtype != np.uint8:
        img1_uint8 = (img1 * 255).astype(np.uint8)
        img2_uint8 = (img2 * 255).astype(np.uint8)
    else:
        img1_uint8 = img1
        img2_uint8 = img2

    # Compute SSIM for grayscale or each channel
    if len(img1.shape) == 2:
        score = cv2.matchTemplate(
            img1_uint8, img2_uint8, cv2.TM_CCOEFF_NORMED)[0, 0]
    else:
        # Use skimage-style SSIM computation for color images
        from skimage.metrics import structural_similarity
        score = structural_similarity(
            img1, img2, multichannel=True, channel_axis=-1)

    return float(score)


def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Absolute Error between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.

    Returns
    -------
    float
        MAE value (lower is better).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    return np.mean(np.abs(img1 - img2))


def nrmse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Normalized Root Mean Squared Error.

    Parameters
    ----------
    img1 : np.ndarray
        First image (reference).
    img2 : np.ndarray
        Second image (test).

    Returns
    -------
    float
        NRMSE value (lower is better).
    """
    mse_value = mse(img1, img2)
    rmse = np.sqrt(mse_value)

    # Normalize by range of reference image
    img_range = img1.max() - img1.min()

    if img_range == 0:
        return 0.0

    return rmse / img_range


def correlation_coefficient(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute correlation coefficient between two images.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1] (higher is better).
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    # Flatten images
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()

    # Compute correlation
    return np.corrcoef(img1_flat, img2_flat)[0, 1]


def snr(img: np.ndarray, noise: np.ndarray = None) -> float:
    """
    Compute Signal-to-Noise Ratio.

    Parameters
    ----------
    img : np.ndarray
        Image (signal + noise).
    noise : np.ndarray, optional
        Noise component. If None, estimated from image, by default None.

    Returns
    -------
    float
        SNR value in dB (higher is better).
    """
    signal_power = np.mean(img ** 2)

    if noise is None:
        # Estimate noise from high-frequency components
        if len(img.shape) == 2:
            laplacian = cv2.Laplacian(img, cv2.CV_32F)
        else:
            laplacian = cv2.Laplacian(cv2.cvtColor(
                (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F) / 255.0

        noise_power = np.var(laplacian)
    else:
        noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)
