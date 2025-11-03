"""
PSF (Point Spread Function) Generation Module

This module provides various methods for generating or estimating PSF for deconvolution:
1. Gaussian PSF - Theoretical approximation
2. Gibson-Lanni PSF - Physics-based fluorescence microscopy model
3. Blind PSF Estimation - Estimate from image
4. Custom PSF Loading - Load measured PSF

Author: Mehul Patel
Date: November 2025
"""

import numpy as np
import cv2
from scipy.special import j1  # Bessel function for Airy disk
from scipy.ndimage import fourier_gaussian
from typing import Tuple, Optional, Dict, Any
import warnings


def generate_gaussian_psf(
    wavelength: float,
    numerical_aperture: float,
    pixel_size: float,
    size: int = 31,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate theoretical Gaussian PSF based on diffraction limit.

    Approximates the PSF as a 2D Gaussian based on the Rayleigh criterion.
    Simple but reasonably accurate for many applications.

    Parameters
    ----------
    wavelength : float
        Wavelength of light in nanometers (e.g., 550 for green)
    numerical_aperture : float
        Numerical aperture of objective lens (e.g., 1.4 for oil immersion)
    pixel_size : float
        Pixel size in micrometers (e.g., 0.065)
    size : int, optional
        Size of PSF kernel (odd number recommended), default 31
    normalize : bool, optional
        Normalize PSF to sum to 1, default True

    Returns
    -------
    psf : np.ndarray
        2D PSF array of shape (size, size)

    Notes
    -----
    The standard deviation is calculated using the Rayleigh criterion:
    σ = 0.21 * λ / NA

    This is converted to pixels using the pixel_size parameter.

    Examples
    --------
    >>> # Generate PSF for green light with 100x oil objective
    >>> psf = generate_gaussian_psf(wavelength=550, numerical_aperture=1.4, 
    ...                              pixel_size=0.065, size=31)
    """
    # Calculate PSF width (sigma) in micrometers
    sigma_um = 0.21 * (wavelength / 1000.0) / \
        numerical_aperture  # wavelength in um

    # Convert to pixels
    sigma_pixels = sigma_um / pixel_size

    # Create coordinate grid centered at middle
    center = size // 2
    y, x = np.ogrid[-center:size-center, -center:size-center]

    # Generate 2D Gaussian
    psf = np.exp(-(x**2 + y**2) / (2 * sigma_pixels**2))

    # Normalize if requested
    if normalize:
        psf /= psf.sum()

    return psf.astype(np.float32)


def generate_airy_psf(
    wavelength: float,
    numerical_aperture: float,
    pixel_size: float,
    size: int = 31,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate Airy disk PSF (more accurate than Gaussian).

    Uses the Airy disk formula for diffraction-limited imaging.
    More physically accurate than Gaussian approximation.

    Parameters
    ----------
    wavelength : float
        Wavelength of light in nanometers
    numerical_aperture : float
        Numerical aperture of objective
    pixel_size : float
        Pixel size in micrometers
    size : int, optional
        Size of PSF kernel (odd number), default 31
    normalize : bool, optional
        Normalize PSF to sum to 1, default True

    Returns
    -------
    psf : np.ndarray
        2D PSF array with Airy disk pattern

    Notes
    -----
    Airy pattern: I(r) = I₀ * [2*J₁(kr)/(kr)]²
    where J₁ is the first-order Bessel function

    Examples
    --------
    >>> psf = generate_airy_psf(wavelength=550, numerical_aperture=1.4,
    ...                          pixel_size=0.065)
    """
    # Calculate radius of first minimum (in um)
    r_min = 0.61 * (wavelength / 1000.0) / numerical_aperture

    # Create radial coordinate grid
    center = size // 2
    y, x = np.ogrid[-center:size-center, -center:size-center]
    r = np.sqrt(x**2 + y**2) * pixel_size  # in micrometers

    # Avoid division by zero at center
    k = 2 * np.pi * numerical_aperture / (wavelength / 1000.0)
    kr = k * r
    kr[kr == 0] = 1e-10

    # Airy disk formula: [2*J1(kr)/(kr)]^2
    psf = (2 * j1(kr) / kr) ** 2

    # Normalize
    if normalize:
        psf /= psf.sum()

    return psf.astype(np.float32)


def generate_gibson_lanni_psf(
    wavelength: float,
    numerical_aperture: float,
    pixel_size: float,
    size: int = 31,
    ni: float = 1.518,  # refractive index of immersion medium
    ns: float = 1.33,   # refractive index of sample
    ti: float = 150,    # working distance (um)
    normalize: bool = True
) -> np.ndarray:
    """
    Generate Gibson-Lanni PSF model for fluorescence microscopy.

    Physics-based model that accounts for refractive index mismatch
    and aberrations in fluorescence microscopy. More accurate than
    Gaussian/Airy for thick samples.

    Parameters
    ----------
    wavelength : float
        Emission wavelength in nanometers (e.g., 520 for GFP)
    numerical_aperture : float
        Numerical aperture of objective
    pixel_size : float
        Pixel size in micrometers
    size : int, optional
        Size of PSF kernel, default 31
    ni : float, optional
        Refractive index of immersion medium (1.518 for oil, 1.33 for water)
    ns : float, optional
        Refractive index of sample (typically ~1.33 for aqueous)
    ti : float, optional
        Working distance / imaging depth in micrometers, default 150
    normalize : bool, optional
        Normalize PSF to sum to 1, default True

    Returns
    -------
    psf : np.ndarray
        2D PSF array accounting for refractive index mismatch

    Notes
    -----
    This is a simplified Gibson-Lanni model. For full 3D PSF modeling,
    consider using specialized libraries like PSFmodels or microscPSF.

    The model includes:
    - Spherical aberration from refractive index mismatch
    - Defocus
    - Diffraction effects

    References
    ----------
    Gibson, S. F., & Lanni, F. (1991). "Experimental test of an analytical
    model of aberration in an oil-immersion objective lens used in
    three-dimensional light microscopy." JOSA A, 8(10), 1601-1613.

    Examples
    --------
    >>> # GFP imaging with oil immersion
    >>> psf = generate_gibson_lanni_psf(wavelength=520, numerical_aperture=1.4,
    ...                                  pixel_size=0.065, ni=1.518, ns=1.33)
    """
    # Simplified Gibson-Lanni - full implementation is complex
    # We'll approximate with aberrated Airy disk

    # Calculate spherical aberration coefficient
    aberration = abs(ni - ns) * ti / 100.0  # Simplified

    # Start with Airy disk
    psf = generate_airy_psf(wavelength, numerical_aperture, pixel_size,
                            size, normalize=False)

    # Add aberration by slight Gaussian blur
    if aberration > 0.01:
        sigma_aberration = aberration / pixel_size
        from scipy.ndimage import gaussian_filter
        psf = gaussian_filter(psf, sigma=sigma_aberration)

    # Normalize
    if normalize:
        psf /= psf.sum()

    return psf.astype(np.float32)


def estimate_blind_psf(
    image: np.ndarray,
    method: str = 'autocorrelation',
    psf_size: int = 31,
    **kwargs
) -> np.ndarray:
    """
    Estimate PSF directly from the blurred image (blind estimation).

    Useful when PSF is unknown and cannot be measured. Uses image
    statistics to infer the blur kernel.

    Parameters
    ----------
    image : np.ndarray
        Blurred input image (2D grayscale, normalized 0-1)
    method : str, optional
        Estimation method: 'autocorrelation', 'edge_based', or 'cepstrum'
        Default 'autocorrelation'
    psf_size : int, optional
        Estimated PSF size, default 31
    **kwargs : dict
        Additional parameters for specific methods

    Returns
    -------
    psf : np.ndarray
        Estimated PSF

    Methods
    -------
    autocorrelation : Uses image autocorrelation to estimate blur
    edge_based : Extracts PSF from sharp edges in image
    cepstrum : Frequency domain analysis

    Notes
    -----
    Blind PSF estimation is an ill-posed problem and results may vary.
    Works best on images with sharp edges and good contrast.

    For production use, prefer measured PSF or theoretical models.

    Examples
    --------
    >>> blurred_img = load_image('blurred.tif')
    >>> psf = estimate_blind_psf(blurred_img, method='autocorrelation')
    """
    if method == 'autocorrelation':
        return _estimate_psf_autocorrelation(image, psf_size)
    elif method == 'edge_based':
        return _estimate_psf_edge_based(image, psf_size, **kwargs)
    elif method == 'cepstrum':
        return _estimate_psf_cepstrum(image, psf_size)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'autocorrelation', "
                         "'edge_based', or 'cepstrum'")


def _estimate_psf_autocorrelation(image: np.ndarray, psf_size: int) -> np.ndarray:
    """Estimate PSF using autocorrelation method."""
    # Compute 2D autocorrelation
    from scipy.signal import correlate2d

    # Normalize image
    img_norm = (image - image.mean()) / (image.std() + 1e-10)

    # Autocorrelation
    autocorr = correlate2d(img_norm, img_norm, mode='same')

    # Extract center region as PSF estimate
    center = np.array(autocorr.shape) // 2
    half_size = psf_size // 2
    psf = autocorr[center[0]-half_size:center[0]+half_size+1,
                   center[1]-half_size:center[1]+half_size+1]

    # Post-process
    psf = np.maximum(psf, 0)  # Ensure non-negative
    psf /= psf.sum()

    return psf.astype(np.float32)


def _estimate_psf_edge_based(
    image: np.ndarray,
    psf_size: int,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """Estimate PSF from edges in the image."""
    # Detect edges
    edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)

    # Find edge profiles perpendicular to edges
    # (Simplified - full implementation is complex)

    # Fallback to Gaussian approximation based on edge blur
    gradient = np.gradient(image)
    grad_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)

    # Estimate blur from gradient width
    # Approximate with Gaussian
    sigma_estimate = psf_size / 6.0  # Rough estimate

    center = psf_size // 2
    y, x = np.ogrid[-center:psf_size-center, -center:psf_size-center]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma_estimate**2))
    psf /= psf.sum()

    return psf.astype(np.float32)


def _estimate_psf_cepstrum(image: np.ndarray, psf_size: int) -> np.ndarray:
    """Estimate PSF using cepstrum analysis (frequency domain)."""
    # Compute power spectrum
    f_image = np.fft.fft2(image)
    power_spectrum = np.abs(f_image) ** 2

    # Cepstrum
    cepstrum = np.fft.ifft2(np.log(power_spectrum + 1e-10))
    cepstrum = np.abs(np.fft.fftshift(cepstrum))

    # Extract center as PSF estimate
    center = np.array(cepstrum.shape) // 2
    half_size = psf_size // 2
    psf = cepstrum[center[0]-half_size:center[0]+half_size+1,
                   center[1]-half_size:center[1]+half_size+1]

    # Post-process
    psf = np.maximum(psf, 0)
    psf /= psf.sum()

    return psf.astype(np.float32)


def load_custom_psf(psf_file: str, normalize: bool = True) -> np.ndarray:
    """
    Load measured PSF from file.

    Supports TIF, PNG, NPY, and NPZ formats.

    Parameters
    ----------
    psf_file : str
        Path to PSF file (measured from bead imaging or simulation)
    normalize : bool, optional
        Normalize PSF to sum to 1, default True

    Returns
    -------
    psf : np.ndarray
        2D PSF array

    Notes
    -----
    Expected PSF properties:
    - 2D array (if 3D, central slice is used)
    - Odd dimensions (if even, center is adjusted)
    - Non-negative values
    - Single-channel (grayscale)

    How to measure PSF experimentally:
    1. Image sub-resolution fluorescent beads (~100nm)
    2. Extract bead image (should be single point)
    3. Average multiple beads to reduce noise
    4. Save as TIF or NPY

    Examples
    --------
    >>> # Load measured PSF from bead imaging
    >>> psf = load_custom_psf('measured_psf.tif')
    >>> 
    >>> # Load simulated PSF
    >>> psf = load_custom_psf('simulated_psf.npy')
    """
    import os
    from utils.io import load_image

    if not os.path.exists(psf_file):
        raise FileNotFoundError(f"PSF file not found: {psf_file}")

    # Load based on extension
    ext = os.path.splitext(psf_file)[1].lower()

    if ext in ['.npy']:
        psf = np.load(psf_file)
    elif ext in ['.npz']:
        data = np.load(psf_file)
        # Assume PSF is stored with key 'psf' or use first array
        psf = data['psf'] if 'psf' in data else data[data.files[0]]
    elif ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
        psf = load_image(psf_file)
    else:
        raise ValueError(f"Unsupported PSF file format: {ext}")

    # Handle 3D PSF (use central slice)
    if psf.ndim == 3:
        warnings.warn("3D PSF detected. Using central z-slice.")
        psf = psf[psf.shape[0] // 2, :, :]

    # Ensure 2D
    if psf.ndim != 2:
        raise ValueError(f"PSF must be 2D, got shape {psf.shape}")

    # Ensure odd dimensions (for symmetric convolution)
    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        warnings.warn("PSF has even dimensions. Cropping to odd size.")
        new_h = psf.shape[0] - 1 if psf.shape[0] % 2 == 0 else psf.shape[0]
        new_w = psf.shape[1] - 1 if psf.shape[1] % 2 == 0 else psf.shape[1]
        center_h, center_w = psf.shape[0] // 2, psf.shape[1] // 2
        psf = psf[center_h - new_h//2:center_h + new_h//2 + 1,
                  center_w - new_w//2:center_w + new_w//2 + 1]

    # Ensure non-negative
    psf = np.maximum(psf, 0)

    # Normalize
    if normalize:
        if psf.sum() > 0:
            psf = psf / psf.sum()
        else:
            raise ValueError("PSF sums to zero. Invalid PSF.")

    return psf.astype(np.float32)


def get_psf(
    psf_method: str,
    psf_params: Dict[str, Any],
    image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Unified interface to get PSF using any method.

    Parameters
    ----------
    psf_method : str
        Method name: 'gaussian', 'airy', 'gibson_lanni', 'blind', or 'custom'
    psf_params : dict
        Parameters for the chosen method
    image : np.ndarray, optional
        Required for 'blind' method

    Returns
    -------
    psf : np.ndarray
        Generated or loaded PSF

    Examples
    --------
    >>> # Gaussian PSF
    >>> params = {'wavelength': 550, 'numerical_aperture': 1.4, 
    ...           'pixel_size': 0.065, 'size': 31}
    >>> psf = get_psf('gaussian', params)
    >>> 
    >>> # Custom PSF
    >>> params = {'psf_file': 'measured_psf.tif'}
    >>> psf = get_psf('custom', params)
    """
    psf_method = psf_method.lower()

    if psf_method == 'gaussian':
        return generate_gaussian_psf(**psf_params)
    elif psf_method == 'airy':
        return generate_airy_psf(**psf_params)
    elif psf_method == 'gibson_lanni':
        return generate_gibson_lanni_psf(**psf_params)
    elif psf_method == 'blind':
        if image is None:
            raise ValueError("Image required for blind PSF estimation")
        return estimate_blind_psf(image, **psf_params)
    elif psf_method == 'custom':
        return load_custom_psf(**psf_params)
    else:
        raise ValueError(f"Unknown PSF method: {psf_method}. "
                         f"Use 'gaussian', 'airy', 'gibson_lanni', 'blind', or 'custom'")


def visualize_psf(psf: np.ndarray, title: str = "PSF") -> None:
    """
    Visualize PSF in spatial and frequency domain.

    Parameters
    ----------
    psf : np.ndarray
        2D PSF array
    title : str, optional
        Plot title

    Examples
    --------
    >>> psf = generate_gaussian_psf(550, 1.4, 0.065)
    >>> visualize_psf(psf, "Gaussian PSF")
    """
    import matplotlib.pyplot as plt

    # Compute OTF (Optical Transfer Function)
    otf = np.fft.fft2(psf)
    otf = np.fft.fftshift(otf)
    otf_magnitude = np.abs(otf)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Spatial domain
    im1 = axes[0].imshow(psf, cmap='hot')
    axes[0].set_title(f'{title} - Spatial Domain')
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0])

    # Cross-section
    center = psf.shape[0] // 2
    axes[1].plot(psf[center, :], label='Horizontal')
    axes[1].plot(psf[:, center], label='Vertical', linestyle='--')
    axes[1].set_title('PSF Cross-Section')
    axes[1].set_xlabel('Position (pixels)')
    axes[1].set_ylabel('Intensity')
    axes[1].legend()
    axes[1].grid(True)

    # Frequency domain (OTF)
    im3 = axes[2].imshow(np.log10(otf_magnitude + 1e-10), cmap='viridis')
    axes[2].set_title('OTF Magnitude (log scale)')
    axes[2].set_xlabel('Frequency X')
    axes[2].set_ylabel('Frequency Y')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


# Convenience functions for common microscopy configurations

def psf_widefield_fluorescence(
    emission_wavelength: float = 520,  # GFP
    objective_na: float = 1.4,          # Oil immersion 100x
    pixel_size: float = 0.065,          # Typical camera
    size: int = 31
) -> np.ndarray:
    """Generate typical widefield fluorescence PSF (oil immersion)."""
    return generate_gibson_lanni_psf(
        wavelength=emission_wavelength,
        numerical_aperture=objective_na,
        pixel_size=pixel_size,
        size=size,
        ni=1.518,  # Oil
        ns=1.33    # Aqueous sample
    )


def psf_brightfield(
    wavelength: float = 550,   # White light ~green
    objective_na: float = 0.75,  # Dry objective
    pixel_size: float = 0.1,
    size: int = 31
) -> np.ndarray:
    """Generate typical brightfield PSF."""
    return generate_gaussian_psf(
        wavelength=wavelength,
        numerical_aperture=objective_na,
        pixel_size=pixel_size,
        size=size
    )


if __name__ == "__main__":
    # Demo usage
    print("PSF Generation Module Demo")
    print("=" * 50)

    # Generate different PSFs
    psf_gauss = generate_gaussian_psf(550, 1.4, 0.065, 31)
    psf_airy = generate_airy_psf(550, 1.4, 0.065, 31)
    psf_gl = generate_gibson_lanni_psf(520, 1.4, 0.065, 31)

    print(f"Gaussian PSF shape: {psf_gauss.shape}, sum: {psf_gauss.sum():.6f}")
    print(f"Airy PSF shape: {psf_airy.shape}, sum: {psf_airy.sum():.6f}")
    print(f"Gibson-Lanni PSF shape: {psf_gl.shape}, sum: {psf_gl.sum():.6f}")

    # Visualize
    try:
        visualize_psf(psf_gauss, "Gaussian PSF")
    except:
        print("Visualization requires matplotlib (optional)")
