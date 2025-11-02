"""Image I/O utilities for loading and saving microscopy images."""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Union
import warnings


def load_image(
    filepath: Union[str, Path],
    as_float: bool = True,
    grayscale: bool = False
) -> np.ndarray:
    """
    Load an image from file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the image file.
    as_float : bool, optional
        Convert to float in [0, 1] range, by default True.
    grayscale : bool, optional
        Convert to grayscale, by default False.

    Returns
    -------
    np.ndarray
        Loaded image array.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    # Load image with appropriate flags
    if grayscale:
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        # Convert BGR to RGB if color image
        if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise IOError(f"Failed to load image: {filepath}")

    # Convert to float if requested
    if as_float:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)

    return img


def save_image(
    img: np.ndarray,
    filepath: Union[str, Path],
    bit_depth: int = 16,
    quality: int = 95
) -> None:
    """
    Save an image to file.

    Parameters
    ----------
    img : np.ndarray
        Image array to save.
    filepath : Union[str, Path]
        Destination file path.
    bit_depth : int, optional
        Bit depth for output (8 or 16), by default 16.
    quality : int, optional
        JPEG quality (0-100), by default 95.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Prepare image for saving
    img_out = img.copy()

    # Clip values to valid range
    img_out = np.clip(img_out, 0, 1)

    # Determine bit depth based on file format
    ext = filepath.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        # JPEG only supports 8-bit
        bit_depth = 8

    # Convert to appropriate dtype
    if bit_depth == 8:
        img_out = (img_out * 255).astype(np.uint8)
    elif bit_depth == 16:
        img_out = (img_out * 65535).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 8 or 16.")

    # Convert RGB to BGR for OpenCV
    if len(img_out.shape) == 3 and img_out.shape[2] == 3:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    # Save with appropriate parameters
    ext = filepath.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        success = cv2.imwrite(str(filepath), img_out, [
                              cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == '.png':
        success = cv2.imwrite(str(filepath), img_out, [
                              cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        success = cv2.imwrite(str(filepath), img_out)

    if not success:
        raise IOError(f"Failed to save image: {filepath}")


def get_image_info(filepath: Union[str, Path]) -> dict:
    """
    Get information about an image file without fully loading it.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the image file.

    Returns
    -------
    dict
        Dictionary containing image metadata.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    # Load just to get info
    img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise IOError(f"Failed to read image: {filepath}")

    info = {
        'filepath': str(filepath),
        'filename': filepath.name,
        'shape': img.shape,
        'dtype': str(img.dtype),
        'channels': img.shape[2] if len(img.shape) == 3 else 1,
        'size_bytes': filepath.stat().st_size,
    }

    return info
