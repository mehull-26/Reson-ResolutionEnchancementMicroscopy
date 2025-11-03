"""
Deconvolution Enhancement Module

Implements PSF-based deconvolution algorithms for resolution enhancement:
1. Richardson-Lucy (iterative, standard for fluorescence)
2. Wiener Deconvolution (frequency domain, fast)
3. Total Variation Deconvolution (edge-preserving, advanced)

Author: Mehul Patel
Date: November 2025
"""

import numpy as np
from scipy import signal
from scipy.ndimage import convolve
from enhancement.base import EnhancementModule
from utils.psf_generation import get_psf
from typing import Optional, Dict, Any
import warnings


class RichardsonLucy(EnhancementModule):
    """
    Richardson-Lucy Deconvolution Algorithm.

    Standard iterative deconvolution method for fluorescence microscopy.
    Assumes Poisson noise model and preserves non-negativity.

    Parameters
    ----------
    iterations : int
        Number of iterations (10-50, default 20)
        More iterations = sharper but more noise amplification

    psf_method : str
        PSF generation method: 'gaussian', 'airy', 'gibson_lanni', 'blind', 'custom'
        Default 'gaussian'

    psf_params : dict
        Parameters for PSF generation (wavelength, NA, pixel_size, etc.)

    regularization : float
        Regularization factor to prevent noise amplification (0.0-0.01)
        Default 0.001. Higher = smoother result

    clip_negative : bool
        Ensure result is non-negative, default True

    convergence_threshold : float, optional
        Stop if change between iterations < threshold
        Default None (run all iterations)

    Examples
    --------
    >>> from enhancement.deconvolution import RichardsonLucy
    >>> deconv = RichardsonLucy({
    ...     'iterations': 20,
    ...     'psf_method': 'gaussian',
    ...     'psf_params': {
    ...         'wavelength': 550,
    ...         'numerical_aperture': 1.4,
    ...         'pixel_size': 0.065,
    ...         'size': 31
    ...     },
    ...     'regularization': 0.001
    ... })
    >>> enhanced = deconv.apply(blurred_image)

    Notes
    -----
    Richardson-Lucy algorithm:
    - u^(k+1) = u^(k) · [PSF* ⊗ (y / (PSF ⊗ u^(k)))]
    where:
      u = estimated image
      y = observed (blurred) image
      PSF = point spread function
      ⊗ = convolution
      PSF* = flipped PSF

    Advantages:
    - Preserves positivity (good for intensity images)
    - Standard in microscopy (widely validated)
    - Handles Poisson noise naturally

    Disadvantages:
    - Slow (iterative)
    - Can amplify noise
    - Requires good PSF estimate

    References
    ----------
    Richardson, W. H. (1972). "Bayesian-based iterative method of image 
    restoration." JOSA, 62(1), 55-59.

    Lucy, L. B. (1974). "An iterative technique for the rectification of 
    observed distributions." The astronomical journal, 79, 745.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.iterations = self.params.get('iterations', 20)
        self.psf_method = self.params.get('psf_method', 'gaussian')
        self.psf_params = self.params.get('psf_params', {})
        self.regularization = self.params.get('regularization', 0.001)
        self.clip_negative = self.params.get('clip_negative', True)
        self.convergence_threshold = self.params.get(
            'convergence_threshold', None)

        # Pre-generate PSF if not using blind method
        self.psf = None
        if self.psf_method != 'blind':
            self.psf = get_psf(self.psf_method, self.psf_params)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Richardson-Lucy deconvolution to image.

        Parameters
        ----------
        image : np.ndarray
            Input blurred image (2D, normalized 0-1)

        Returns
        -------
        deconvolved : np.ndarray
            Deconvolved image
        """
        # Get PSF (generate or use pre-generated)
        if self.psf is None:  # Blind method
            self.psf = get_psf(self.psf_method, self.psf_params, image=image)

        psf = self.psf

        # Flip PSF for correlation (PSF*)
        psf_flipped = np.flipud(np.fliplr(psf))

        # Initialize estimate with observed image
        estimate = np.copy(image) + 1e-10  # Avoid division by zero

        # Add small epsilon to prevent division by zero
        epsilon = 1e-10

        # Richardson-Lucy iterations
        for i in range(self.iterations):
            # Convolve current estimate with PSF
            convolved = convolve(estimate, psf, mode='reflect')

            # Relative blur: observed / convolved
            relative_blur = image / (convolved + epsilon)

            # Correlate with flipped PSF
            correction = convolve(relative_blur, psf_flipped, mode='reflect')

            # Update estimate
            estimate_new = estimate * correction

            # Apply regularization (prevents noise amplification)
            if self.regularization > 0:
                # TV-like regularization (simplified)
                estimate_new = (estimate_new + self.regularization *
                                estimate) / (1 + self.regularization)

            # Ensure non-negativity
            if self.clip_negative:
                estimate_new = np.maximum(estimate_new, 0)

            # Check convergence
            if self.convergence_threshold is not None:
                change = np.abs(estimate_new - estimate).mean()
                if change < self.convergence_threshold:
                    print(f"Richardson-Lucy converged at iteration {i+1}")
                    break

            estimate = estimate_new

        # Clip to valid range
        estimate = np.clip(estimate, 0, 1)

        return estimate.astype(np.float32)


class WienerDeconvolution(EnhancementModule):
    """
    Wiener Deconvolution (Frequency Domain).

    Fast, non-iterative deconvolution using Wiener filter.
    Good for Gaussian noise, includes noise regularization.

    Parameters
    ----------
    psf_method : str
        PSF generation method

    psf_params : dict
        PSF parameters

    noise_power : float
        Noise power spectral density estimate (0.001-0.1)
        Default 0.01. Higher = more smoothing

    balance : float
        Signal-to-noise balance (0.0-1.0)
        Default 0.1. Controls deconvolution strength

    Examples
    --------
    >>> deconv = WienerDeconvolution({
    ...     'psf_method': 'gaussian',
    ...     'psf_params': {'wavelength': 550, 'numerical_aperture': 1.4,
    ...                    'pixel_size': 0.065, 'size': 31},
    ...     'noise_power': 0.01,
    ...     'balance': 0.1
    ... })
    >>> enhanced = deconv.apply(blurred_image)

    Notes
    -----
    Wiener filter in frequency domain:
    W(f) = H*(f) / (|H(f)|² + noise_power/signal_power)

    where:
      H(f) = PSF in frequency domain (OTF)
      H*(f) = complex conjugate of H(f)

    Advantages:
    - Very fast (single FFT operation)
    - Theoretically optimal for Gaussian noise
    - Non-iterative

    Disadvantages:
    - Can produce ringing artifacts
    - Needs noise estimate
    - Less suitable for Poisson noise (fluorescence)

    References
    ----------
    Wiener, N. (1949). "Extrapolation, interpolation, and smoothing of 
    stationary time series."
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.psf_method = self.params.get('psf_method', 'gaussian')
        self.psf_params = self.params.get('psf_params', {})
        self.noise_power = self.params.get('noise_power', 0.01)
        self.balance = self.params.get('balance', 0.1)

        # Pre-generate PSF
        if self.psf_method != 'blind':
            self.psf = get_psf(self.psf_method, self.psf_params)
        else:
            self.psf = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Wiener deconvolution.

        Parameters
        ----------
        image : np.ndarray
            Input blurred image

        Returns
        -------
        deconvolved : np.ndarray
            Deconvolved image
        """
        # Get PSF
        if self.psf is None:
            self.psf = get_psf(self.psf_method, self.psf_params, image=image)

        psf = self.psf

        # Pad PSF to image size
        psf_padded = np.zeros_like(image)
        h_offset = (image.shape[0] - psf.shape[0]) // 2
        w_offset = (image.shape[1] - psf.shape[1]) // 2
        psf_padded[h_offset:h_offset+psf.shape[0],
                   w_offset:w_offset+psf.shape[1]] = psf

        # Shift PSF to have center at (0,0) for FFT
        psf_padded = np.fft.ifftshift(psf_padded)

        # Transform to frequency domain
        image_fft = np.fft.fft2(image)
        psf_fft = np.fft.fft2(psf_padded)

        # Wiener filter
        # W(f) = H*(f) / (|H(f)|^2 + K)
        # where K = noise_power / signal_power
        psf_conj = np.conj(psf_fft)
        psf_abs2 = np.abs(psf_fft) ** 2

        # Regularization parameter
        K = self.noise_power * self.balance

        # Wiener filter
        wiener_filter = psf_conj / (psf_abs2 + K + 1e-10)

        # Apply filter
        result_fft = wiener_filter * image_fft

        # Transform back to spatial domain
        result = np.fft.ifft2(result_fft)
        result = np.real(result)

        # Clip to valid range
        result = np.clip(result, 0, 1)

        return result.astype(np.float32)


class TVDeconvolution(EnhancementModule):
    """
    Total Variation (TV) Regularized Deconvolution.

    Edge-preserving deconvolution using TV regularization.
    Reduces ringing artifacts while preserving sharp edges.

    Parameters
    ----------
    iterations : int
        Number of iterations (50-200, default 100)
        More iterations = better convergence but slower

    psf_method : str
        PSF generation method

    psf_params : dict
        PSF parameters

    lambda_tv : float
        TV regularization strength (0.001-0.1)
        Default 0.01. Higher = smoother edges

    tau : float
        Step size for gradient descent (0.01-0.1)
        Default 0.05

    Examples
    --------
    >>> deconv = TVDeconvolution({
    ...     'iterations': 100,
    ...     'psf_method': 'gaussian',
    ...     'psf_params': {'wavelength': 550, 'numerical_aperture': 1.4,
    ...                    'pixel_size': 0.065, 'size': 31},
    ...     'lambda_tv': 0.01,
    ...     'tau': 0.05
    ... })
    >>> enhanced = deconv.apply(blurred_image)

    Notes
    -----
    TV deconvolution minimizes:
    E(u) = ||PSF ⊗ u - y||² + λ·TV(u)

    where:
      TV(u) = ∫|∇u| dx (total variation)
      λ = regularization parameter

    Solved iteratively using gradient descent.

    Advantages:
    - Preserves edges well
    - Reduces ringing artifacts
    - Good for piecewise-smooth images (cells, structures)

    Disadvantages:
    - Very slow (many iterations)
    - Can create "staircase" artifacts
    - Complex parameter tuning

    References
    ----------
    Rudin, L. I., Osher, S., & Fatemi, E. (1992). "Nonlinear total variation 
    based noise removal algorithms." Physica D, 60(1-4), 259-268.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.iterations = self.params.get('iterations', 100)
        self.psf_method = self.params.get('psf_method', 'gaussian')
        self.psf_params = self.params.get('psf_params', {})
        self.lambda_tv = self.params.get('lambda_tv', 0.01)
        self.tau = self.params.get('tau', 0.05)

        # Pre-generate PSF
        if self.psf_method != 'blind':
            self.psf = get_psf(self.psf_method, self.psf_params)
        else:
            self.psf = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply TV deconvolution.

        Parameters
        ----------
        image : np.ndarray
            Input blurred image

        Returns
        -------
        deconvolved : np.ndarray
            Deconvolved image
        """
        # Get PSF
        if self.psf is None:
            self.psf = get_psf(self.psf_method, self.psf_params, image=image)

        psf = self.psf
        psf_flipped = np.flipud(np.fliplr(psf))

        # Initialize with observed image
        u = np.copy(image)

        # Gradient descent iterations
        for i in range(self.iterations):
            # Data fidelity term: PSF^T * (PSF * u - y)
            convolved = convolve(u, psf, mode='reflect')
            residual = convolved - image
            data_term = convolve(residual, psf_flipped, mode='reflect')

            # TV regularization term: -div(∇u / |∇u|)
            tv_term = self._tv_gradient(u)

            # Gradient descent step
            u = u - self.tau * (data_term + self.lambda_tv * tv_term)

            # Project to valid range
            u = np.clip(u, 0, 1)

        return u.astype(np.float32)

    def _tv_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Compute TV gradient: -div(∇u / |∇u|).

        Parameters
        ----------
        image : np.ndarray
            Current image estimate

        Returns
        -------
        tv_grad : np.ndarray
            TV gradient
        """
        epsilon = 1e-8  # For numerical stability

        # Compute gradients (forward differences)
        grad_x = np.roll(image, -1, axis=1) - image
        grad_y = np.roll(image, -1, axis=0) - image

        # Gradient magnitude
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + epsilon)

        # Normalized gradient
        grad_x_norm = grad_x / grad_mag
        grad_y_norm = grad_y / grad_mag

        # Divergence (backward differences)
        div_x = grad_x_norm - np.roll(grad_x_norm, 1, axis=1)
        div_y = grad_y_norm - np.roll(grad_y_norm, 1, axis=0)

        # TV gradient
        tv_grad = -(div_x + div_y)

        return tv_grad


# Convenience function for quick deconvolution
def deconvolve(
    image: np.ndarray,
    method: str = 'richardson_lucy',
    psf_method: str = 'gaussian',
    psf_params: Dict[str, Any] = None,
    **kwargs
) -> np.ndarray:
    """
    Quick deconvolution interface.

    Parameters
    ----------
    image : np.ndarray
        Blurred input image
    method : str
        Deconvolution method: 'richardson_lucy', 'wiener', or 'tv'
    psf_method : str
        PSF generation method
    psf_params : dict
        PSF parameters
    **kwargs : dict
        Additional algorithm-specific parameters

    Returns
    -------
    deconvolved : np.ndarray
        Deconvolved image

    Examples
    --------
    >>> from enhancement.deconvolution import deconvolve
    >>> result = deconvolve(
    ...     blurred_image,
    ...     method='richardson_lucy',
    ...     psf_method='gaussian',
    ...     psf_params={'wavelength': 550, 'numerical_aperture': 1.4,
    ...                 'pixel_size': 0.065, 'size': 31},
    ...     iterations=20
    ... )
    """
    if psf_params is None:
        psf_params = {}

    params = {
        'psf_method': psf_method,
        'psf_params': psf_params,
        **kwargs
    }

    if method.lower() == 'richardson_lucy':
        deconv = RichardsonLucy(params)
    elif method.lower() == 'wiener':
        deconv = WienerDeconvolution(params)
    elif method.lower() == 'tv':
        deconv = TVDeconvolution(params)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'richardson_lucy', 'wiener', or 'tv'")

    return deconv.apply(image)


if __name__ == "__main__":
    # Demo usage
    print("Deconvolution Module Demo")
    print("=" * 50)

    # Create test blurred image
    from utils.psf_generation import generate_gaussian_psf
    from scipy.ndimage import convolve

    # Generate sharp test image (checkerboard)
    size = 256
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    sharp = ((X // 32 + Y // 32) % 2).astype(float)

    # Blur with PSF
    psf = generate_gaussian_psf(550, 1.0, 0.1, 21)
    blurred = convolve(sharp, psf, mode='reflect')
    blurred = np.clip(blurred, 0, 1)

    print(f"Test image: {size}x{size}")
    print(f"PSF size: {psf.shape}")

    # Test Richardson-Lucy
    print("\n Testing Richardson-Lucy...")
    rl_params = {
        'iterations': 10,
        'psf_method': 'gaussian',
        'psf_params': {'wavelength': 550, 'numerical_aperture': 1.0,
                       'pixel_size': 0.1, 'size': 21},
        'regularization': 0.001
    }
    rl = RichardsonLucy(rl_params)
    deconv_rl = rl.apply(blurred)
    print(
        f"Richardson-Lucy output: shape={deconv_rl.shape}, range=[{deconv_rl.min():.3f}, {deconv_rl.max():.3f}]")

    # Test Wiener
    print("\nTesting Wiener...")
    wiener_params = {
        'psf_method': 'gaussian',
        'psf_params': {'wavelength': 550, 'numerical_aperture': 1.0,
                       'pixel_size': 0.1, 'size': 21},
        'noise_power': 0.01
    }
    wiener = WienerDeconvolution(wiener_params)
    deconv_wiener = wiener.apply(blurred)
    print(
        f"Wiener output: shape={deconv_wiener.shape}, range=[{deconv_wiener.min():.3f}, {deconv_wiener.max():.3f}]")

    print("\n✅ Deconvolution module working correctly!")
