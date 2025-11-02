"""Sharpness and focus metrics."""

import numpy as np
import cv2
from scipy.ndimage import sobel, laplace


def gradient_sharpness(img: np.ndarray) -> float:
    """
    Compute gradient-based sharpness measure.

    Uses the average magnitude of image gradients as a sharpness indicator.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    float
        Sharpness value (higher is sharper).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = img

    # Compute gradients
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Return mean gradient magnitude
    return float(np.mean(grad_mag))


def laplacian_variance(img: np.ndarray) -> float:
    """
    Compute Laplacian variance as a focus/sharpness measure.

    Higher variance indicates sharper focus.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    float
        Laplacian variance (higher is sharper).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8)

    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Return variance
    return float(laplacian.var())


def brenner_sharpness(img: np.ndarray) -> float:
    """
    Compute Brenner sharpness measure.

    Based on squared gradients with a spacing of 2 pixels.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    float
        Brenner sharpness value (higher is sharper).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = (img * 255).astype(np.float32)

    # Compute differences with spacing of 2
    diff_x = gray[:, 2:] - gray[:, :-2]
    diff_y = gray[2:, :] - gray[:-2, :]

    # Brenner sharpness
    sharpness = np.sum(diff_x ** 2) + np.sum(diff_y ** 2)

    return float(sharpness / gray.size)


def tenengrad_sharpness(img: np.ndarray, threshold: float = 0) -> float:
    """
    Compute Tenengrad sharpness measure.

    Based on Sobel operator gradient magnitude.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    threshold : float, optional
        Threshold for gradient magnitude, by default 0.

    Returns
    -------
    float
        Tenengrad sharpness value (higher is sharper).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = (img * 255).astype(np.float32)

    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Apply threshold
    grad_mag[grad_mag < threshold] = 0

    # Tenengrad measure
    sharpness = np.sum(grad_mag ** 2)

    return float(sharpness / gray.size)


def variance_sharpness(img: np.ndarray) -> float:
    """
    Compute image variance as a simple sharpness measure.

    Higher variance typically indicates sharper images.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    float
        Image variance (higher is sharper).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = img

    return float(np.var(gray))


def entropy_sharpness(img: np.ndarray, bins: int = 256) -> float:
    """
    Compute image entropy as a complexity/sharpness measure.

    Higher entropy can indicate more detail/sharpness.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    bins : int, optional
        Number of histogram bins, by default 256.

    Returns
    -------
    float
        Image entropy (higher indicates more information).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8)

    # Compute histogram
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))

    # Normalize to probabilities
    hist = hist.astype(np.float32) / hist.sum()

    # Remove zeros
    hist = hist[hist > 0]

    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def frequency_domain_sharpness(img: np.ndarray, threshold_percentile: float = 90) -> float:
    """
    Compute sharpness in frequency domain.

    Measures the amount of high-frequency content.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    threshold_percentile : float, optional
        Percentile threshold for high frequencies, by default 90.

    Returns
    -------
    float
        High-frequency content measure (higher is sharper).
    """
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = (img * 255).astype(np.float32)

    # Compute FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Compute threshold
    threshold = np.percentile(magnitude, threshold_percentile)

    # Count high-frequency components
    high_freq_content = np.sum(magnitude > threshold)

    return float(high_freq_content / magnitude.size)
