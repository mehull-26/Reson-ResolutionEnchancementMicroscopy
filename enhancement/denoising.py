"""Denoising enhancement modules."""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from typing import Dict, Any

from .base import EnhancementModule


class BilateralFilter(EnhancementModule):
    """
    Bilateral filtering for edge-preserving denoising.

    Reduces noise while preserving edges by considering both spatial
    and intensity similarities.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.d = self.params.get('d', 9)
        self.sigma_color = self.params.get('sigma_color', 75)
        self.sigma_space = self.params.get('sigma_space', 75)

        if self.d <= 0:
            raise ValueError("d must be positive")
        if self.sigma_color <= 0:
            raise ValueError("sigma_color must be positive")
        if self.sigma_space <= 0:
            raise ValueError("sigma_space must be positive")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply bilateral filter."""
        # Convert to uint8 for OpenCV
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(
            img_uint8,
            d=self.d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )

        # Convert back to float
        return filtered.astype(np.float32) / 255.0


class NonLocalMeans(EnhancementModule):
    """
    Non-Local Means denoising.

    Exploits patch similarity across the image for effective denoising.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.h = self.params.get('h', 10)
        self.template_window_size = self.params.get('template_window_size', 7)
        self.search_window_size = self.params.get('search_window_size', 21)

        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.template_window_size % 2 == 0:
            raise ValueError("template_window_size must be odd")
        if self.search_window_size % 2 == 0:
            raise ValueError("search_window_size must be odd")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means denoising."""
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply NLM denoising
        if len(img.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(
                img_uint8,
                h=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )
        else:
            denoised = cv2.fastNlMeansDenoisingColored(
                img_uint8,
                h=self.h,
                hColor=self.h,
                templateWindowSize=self.template_window_size,
                searchWindowSize=self.search_window_size
            )

        # Convert back to float
        return denoised.astype(np.float32) / 255.0


class GaussianDenoising(EnhancementModule):
    """
    Gaussian blur for basic denoising.

    Simple but effective for mild noise reduction.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.sigma = self.params.get('sigma', 1.0)

        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian denoising."""
        if len(img.shape) == 2:
            denoised = gaussian_filter(img, sigma=self.sigma)
        else:
            denoised = np.stack([
                gaussian_filter(img[..., i], sigma=self.sigma)
                for i in range(img.shape[2])
            ], axis=-1)

        return denoised


class MedianFilter(EnhancementModule):
    """
    Median filtering for impulse noise removal.

    Effective for salt-and-pepper noise while preserving edges.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.ksize = self.params.get('ksize', 5)

        if self.ksize <= 0 or self.ksize % 2 == 0:
            raise ValueError("ksize must be positive and odd")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply median filter."""
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)

        # Apply median filter
        filtered = cv2.medianBlur(img_uint8, self.ksize)

        # Convert back to float
        return filtered.astype(np.float32) / 255.0


class AnisotropicDiffusion(EnhancementModule):
    """
    Anisotropic diffusion for edge-preserving smoothing.

    Applies diffusion that adapts to local image structure.
    """

    def validate_params(self) -> None:
        """Validate parameters."""
        self.alpha = self.params.get('alpha', 0.2)
        self.kappa = self.params.get('kappa', 50)
        self.iterations = self.params.get('iterations', 10)

        if self.alpha <= 0 or self.alpha > 0.25:
            raise ValueError("alpha must be in (0, 0.25]")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply anisotropic diffusion."""
        # Work on float image directly
        output = img.copy()

        # Handle grayscale and color separately
        if len(img.shape) == 2:
            output = self._diffuse_channel(output)
        else:
            for i in range(img.shape[2]):
                output[..., i] = self._diffuse_channel(output[..., i])

        return np.clip(output, 0, 1)

    def _diffuse_channel(self, channel: np.ndarray) -> np.ndarray:
        """Apply diffusion to a single channel."""
        img = channel.copy()

        for _ in range(self.iterations):
            # Compute gradients
            grad_n = np.roll(img, -1, axis=0) - img
            grad_s = np.roll(img, 1, axis=0) - img
            grad_e = np.roll(img, -1, axis=1) - img
            grad_w = np.roll(img, 1, axis=1) - img

            # Compute conduction coefficients (Perona-Malik)
            c_n = np.exp(-(grad_n / self.kappa) ** 2)
            c_s = np.exp(-(grad_s / self.kappa) ** 2)
            c_e = np.exp(-(grad_e / self.kappa) ** 2)
            c_w = np.exp(-(grad_w / self.kappa) ** 2)

            # Update image
            img += self.alpha * (
                c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w
            )

        return img
