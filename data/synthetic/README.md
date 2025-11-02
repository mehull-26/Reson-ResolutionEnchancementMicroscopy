# Synthetic Dataset Metadata

Generated: 512x512 images

## Patterns:
- circles: Random circles of various sizes
- cells: Cell-like structures (ellipses with nuclei)
- lines: Fiber-like linear structures
- grid: Regular grid pattern
- mixed: Combination of multiple features
- siemens_star: Resolution test pattern

## Degradations:

### Blurred:
- PSF: Gaussian, size=15x15
- Simulates: Diffraction-limited imaging

### Noisy:
- gaussian: sigma=0.05 (5% noise)
- poisson: Photon counting noise (scale=100)
- saltpepper: 1% pixels corrupted

### Blurred+Noisy:
- PSF blur + Gaussian noise (sigma=0.03)
- Most realistic microscopy scenario

## Usage:
1. Test denoising: Use 'noisy' images, compare to 'ground_truth'
2. Test sharpening: Use 'blurred' images, compare to 'ground_truth'
3. Test full pipeline: Use 'blurred_noisy', compare to 'ground_truth'

## Metrics to compute:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- Sharpness measures (Laplacian variance, Tenengrad)
