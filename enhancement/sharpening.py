"""Sharpening enhancement modules."""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import Dict, Any

from .base import EnhancementModule


class UnsharpMasking(EnhancementModule):
    """
    Unsharp masking for image sharpening.

    Enhances edges by subtracting a blurred version of the image.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.sigma = self.params.get('sigma', 1.0)
        self.amount = self.params.get('amount', 1.5)
        self.threshold = self.params.get('threshold', 0)

        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.amount < 0:
            raise ValueError("amount must be non-negative")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply unsharp masking."""
        # Create blurred version
        if len(img.shape) == 2:
            blurred = gaussian_filter(img, sigma=self.sigma)
        else:
            blurred = np.stack([
                gaussian_filter(img[..., i], sigma=self.sigma)
                for i in range(img.shape[2])
            ], axis=-1)

        # Create sharpening mask
        mask = img - blurred

        # Apply threshold if specified
        if self.threshold > 0:
            mask = np.where(np.abs(mask) >= self.threshold, mask, 0)

        # Apply sharpening
        sharpened = img + self.amount * mask

        # Clip to valid range
        return np.clip(sharpened, 0, 1)


class GuidedFilter(EnhancementModule):
    """
    Edge-aware sharpening using guided filter.

    Preserves edges while enhancing details.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.radius = self.params.get('radius', 4)
        self.eps = self.params.get('eps', 0.01)
        self.amount = self.params.get('amount', 1.5)

        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply guided filter sharpening."""
        # Convert to uint8 for OpenCV guided filter
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply guided filter (acts as edge-preserving blur)
        if len(img.shape) == 2:
            filtered = cv2.ximgproc.guidedFilter(
                guide=img_uint8,
                src=img_uint8,
                radius=self.radius,
                eps=self.eps
            )
        else:
            filtered = cv2.ximgproc.guidedFilter(
                guide=img_uint8,
                src=img_uint8,
                radius=self.radius,
                eps=self.eps
            )

        # Convert back to float
        filtered = filtered.astype(np.float32) / 255.0

        # Create detail layer
        detail = img - filtered

        # Enhance details
        sharpened = img + self.amount * detail

        return np.clip(sharpened, 0, 1)


class BilateralSharpening(EnhancementModule):
    """
    Edge-aware sharpening using bilateral filter.

    Uses bilateral filter for edge-preserving smoothing, then enhances details.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.d = self.params.get('d', 9)
        self.sigma_color = self.params.get('sigma_color', 75)
        self.sigma_space = self.params.get('sigma_space', 75)
        self.amount = self.params.get('amount', 1.5)

        if self.d <= 0:
            raise ValueError("d must be positive")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply bilateral filter sharpening."""
        # Convert to uint8 for bilateral filter
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            img_uint8,
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )

        # Convert back to float
        filtered = filtered.astype(np.float32) / 255.0

        # Create detail layer
        detail = img - filtered

        # Enhance details
        sharpened = img + self.amount * detail

        return np.clip(sharpened, 0, 1)


class LaplacianSharpening(EnhancementModule):
    """
    Sharpening using Laplacian operator.

    Enhances edges using the second derivative (Laplacian).
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.ksize = self.params.get('ksize', 3)
        self.amount = self.params.get('amount', 1.0)

        if self.ksize not in [1, 3, 5, 7]:
            raise ValueError("ksize must be 1, 3, 5, or 7")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply Laplacian sharpening."""
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply Laplacian
        if len(img.shape) == 2:
            laplacian = cv2.Laplacian(img_uint8, cv2.CV_32F, ksize=self.ksize)
        else:
            laplacian = np.stack([
                cv2.Laplacian(img_uint8[..., i], cv2.CV_32F, ksize=self.ksize)
                for i in range(img.shape[2])
            ], axis=-1)

        # Normalize Laplacian
        laplacian = laplacian / 255.0

        # Add Laplacian to original
        sharpened = img + self.amount * laplacian

        return np.clip(sharpened, 0, 1)
