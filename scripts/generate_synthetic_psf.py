"""
Generate Synthetic PSF-Blurred Images for Testing Deconvolution

This script creates test images by:
1. Generating clean synthetic images (various patterns)
2. Blurring them with different PSF models
3. Adding optional noise (Gaussian, Poisson)
4. Saving blurred images, ground truth, and PSF files

Author: Mehul Patel
Date: November 2025
"""

from scipy.ndimage import convolve
from utils.io import save_image
from utils.psf_generation import (
    generate_gaussian_psf, generate_airy_psf, generate_gibson_lanni_psf
)
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

sys.path.append(str(Path(__file__).parent.parent))


def generate_test_patterns(size=512):
    """
    Generate various test patterns for PSF blur testing.

    Returns
    -------
    patterns : dict
        Dictionary of test patterns
    """
    patterns = {}

    # 1. Checkerboard (sharp edges)
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    checkerboard = ((X // 64 + Y // 64) % 2).astype(float)
    patterns['checkerboard'] = checkerboard

    # 2. Radial star pattern (high frequency)
    center_x, center_y = size // 2, size // 2
    angles = 16
    star = np.zeros((size, size))
    for i in range(angles):
        angle = i * np.pi / angles
        for r in range(size // 2):
            x = int(center_x + r * np.cos(angle))
            y = int(center_y + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                star[y, x] = 1.0
    # Dilate lines
    kernel = np.ones((3, 3), np.uint8)
    star = cv2.dilate(star.astype(np.uint8), kernel,
                      iterations=2).astype(float)
    patterns['star'] = star

    # 3. Resolution target (Siemens star)
    theta = np.arctan2(Y - center_y, X - center_x)
    r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    num_spokes = 24
    siemens = ((np.sin(num_spokes * theta) > 0) &
               (r < size // 2 - 10)).astype(float)
    patterns['siemens_star'] = siemens

    # 4. Gaussian dots (simulated fluorescent beads)
    dots = np.zeros((size, size))
    np.random.seed(42)
    num_dots = 50
    for _ in range(num_dots):
        cx = np.random.randint(50, size - 50)
        cy = np.random.randint(50, size - 50)
        sigma = 3.0
        for dx in range(-15, 16):
            for dy in range(-15, 16):
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    dots[y, x] += np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    dots = np.clip(dots, 0, 1)
    patterns['fluorescent_beads'] = dots

    # 5. Text/letters (complex shapes)
    text_img = np.zeros((size, size), dtype=np.uint8)
    cv2.putText(text_img, 'DECONV', (50, size//2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 4)
    patterns['text'] = text_img.astype(float) / 255.0

    # 6. Cells-like structures
    cells = np.zeros((size, size))
    np.random.seed(123)
    num_cells = 20
    for _ in range(num_cells):
        cx = np.random.randint(80, size - 80)
        cy = np.random.randint(80, size - 80)
        radius = np.random.randint(20, 40)
        cv2.circle(cells, (cx, cy), radius, 1.0, -1)
        # Add nucleus
        cv2.circle(cells, (cx, cy), radius//3, 0.5, -1)
    patterns['cells'] = np.clip(cells, 0, 1)

    # 7. Grid pattern (multiple scales)
    grid = np.zeros((size, size))
    # Coarse grid
    for i in range(0, size, 64):
        grid[i, :] = 1.0
        grid[:, i] = 1.0
    # Fine grid
    for i in range(0, size, 16):
        grid[i, :] = 0.5
        grid[:, i] = 0.5
    patterns['grid'] = grid

    # 8. Random noise pattern (texture)
    np.random.seed(456)
    noise_pattern = np.random.rand(size, size)
    # Smooth it slightly
    noise_pattern = cv2.GaussianBlur(noise_pattern, (0, 0), 2.0)
    noise_pattern = (noise_pattern - noise_pattern.min()) / \
        (noise_pattern.max() - noise_pattern.min())
    patterns['texture'] = noise_pattern

    return patterns


def blur_with_psf(image, psf):
    """
    Blur image with PSF (convolution).

    Parameters
    ----------
    image : np.ndarray
        Clean image
    psf : np.ndarray
        Point spread function

    Returns
    -------
    blurred : np.ndarray
        Blurred image
    """
    blurred = convolve(image, psf, mode='reflect')
    blurred = np.clip(blurred, 0, 1)
    return blurred


def add_noise(image, noise_type='gaussian', noise_level=0.02):
    """
    Add noise to image.

    Parameters
    ----------
    image : np.ndarray
        Clean image (0-1)
    noise_type : str
        'gaussian' or 'poisson'
    noise_level : float
        Noise strength (for Gaussian: std, for Poisson: scaling factor)

    Returns
    -------
    noisy : np.ndarray
        Noisy image
    """
    if noise_type == 'gaussian':
        noise = np.random.randn(*image.shape) * noise_level
        noisy = image + noise
    elif noise_type == 'poisson':
        # Poisson noise (scale up, add noise, scale down)
        vals = np.maximum(image, 0)
        vals = vals / noise_level
        noisy = np.random.poisson(vals).astype(float)
        noisy = noisy * noise_level
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    noisy = np.clip(noisy, 0, 1)
    return noisy


def main():
    """Generate all synthetic test data."""
    print("Generating Synthetic PSF-Blurred Test Data")
    print("=" * 60)

    # Output directory
    output_dir = Path('data/synthetic_psf')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test patterns
    print("\n1. Generating test patterns...")
    patterns = generate_test_patterns(size=512)
    print(f"   Generated {len(patterns)} patterns")

    # PSF configurations
    psf_configs = {
        'gaussian_mild': {
            'method': generate_gaussian_psf,
            'params': {'wavelength': 550, 'numerical_aperture': 1.0,
                       'pixel_size': 0.1, 'size': 21}
        },
        'gaussian_strong': {
            'method': generate_gaussian_psf,
            'params': {'wavelength': 550, 'numerical_aperture': 0.5,
                       'pixel_size': 0.1, 'size': 31}
        },
        'airy': {
            'method': generate_airy_psf,
            'params': {'wavelength': 550, 'numerical_aperture': 1.4,
                       'pixel_size': 0.065, 'size': 31}
        },
        'fluorescence': {
            'method': generate_gibson_lanni_psf,
            'params': {'wavelength': 520, 'numerical_aperture': 1.4,
                       'pixel_size': 0.065, 'size': 41,
                       'ni': 1.518, 'ns': 1.33, 'ti': 150}
        },
    }

    # Generate PSFs
    print("\n2. Generating PSFs...")
    psfs = {}
    psf_dir = output_dir / 'psfs'
    psf_dir.mkdir(exist_ok=True)

    for psf_name, config in psf_configs.items():
        print(f"   - {psf_name}")
        psf = config['method'](**config['params'])
        psfs[psf_name] = psf

        # Save PSF as NPY and TIF
        np.save(psf_dir / f'{psf_name}.npy', psf)
        # Generate blurred images
        save_image(psf, psf_dir / f'{psf_name}.tif', bit_depth=16)
    print("\n3. Generating blurred images...")

    # Select representative patterns for each PSF
    test_cases = [
        # Pattern, PSF, Noise type, Noise level
        ('checkerboard', 'gaussian_mild', None, 0),
        ('checkerboard', 'gaussian_strong', None, 0),
        ('star', 'airy', None, 0),
        ('siemens_star', 'gaussian_mild', None, 0),
        ('fluorescent_beads', 'fluorescence', 'poisson', 0.05),
        ('fluorescent_beads', 'fluorescence', 'gaussian', 0.02),
        ('text', 'gaussian_strong', 'gaussian', 0.01),
        ('cells', 'fluorescence', 'poisson', 0.03),
        ('grid', 'airy', None, 0),
        ('texture', 'gaussian_mild', 'gaussian', 0.02),
    ]

    count = 0
    for pattern_name, psf_name, noise_type, noise_level in test_cases:
        print(f"   - {pattern_name} + {psf_name}" +
              (f" + {noise_type} noise" if noise_type else ""))

        # Get pattern and PSF
        pattern = patterns[pattern_name]
        psf = psfs[psf_name]

        # Blur
        blurred = blur_with_psf(pattern, psf)

        # Add noise if specified
        if noise_type:
            blurred = add_noise(blurred, noise_type, noise_level)

        # Create test case directory
        case_name = f"{pattern_name}_{psf_name}"
        if noise_type:
            case_name += f"_{noise_type}{int(noise_level*100)}"

        case_dir = output_dir / case_name
        case_dir.mkdir(exist_ok=True)

        # Save images
        save_image(blurred, case_dir / 'blurred.tif', bit_depth=16)
        save_image(pattern, case_dir / 'ground_truth.tif', bit_depth=16)

        # Copy PSF to case directory for convenience
        np.save(case_dir / 'psf.npy', psf)
        save_image(psf, case_dir / 'psf.tif',
                   bit_depth=16)        # Save metadata
        with open(case_dir / 'metadata.txt', 'w') as f:
            f.write(f"Pattern: {pattern_name}\n")
            f.write(f"PSF: {psf_name}\n")
            f.write(f"PSF params: {psf_configs[psf_name]['params']}\n")
            if noise_type:
                f.write(f"Noise: {noise_type}, level={noise_level}\n")
            f.write(f"Image size: {pattern.shape}\n")

        count += 1

    print(f"\n✅ Generated {count} test cases in {output_dir}/")
    print(f"✅ Generated {len(psfs)} PSF files in {psf_dir}/")

    # Create README
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("Synthetic PSF-Blurred Test Data\n")
        f.write("=" * 60 + "\n\n")
        f.write("This directory contains synthetic test images for deconvolution.\n\n")
        f.write("Structure:\n")
        f.write("  - Each test case has its own folder\n")
        f.write("  - Each folder contains:\n")
        f.write("    * blurred.tif - PSF-blurred image (input for deconvolution)\n")
        f.write("    * ground_truth.tif - Original clean image (for comparison)\n")
        f.write("    * psf.npy/.tif - PSF used for blurring\n")
        f.write("    * metadata.txt - Description of test case\n\n")
        f.write(f"Generated {count} test cases\n\n")
        f.write("Test with:\n")
        f.write("  python main.py -i data/synthetic_psf/<case_name>/blurred.tif ")
        f.write("-g data/synthetic_psf/<case_name>/ground_truth.tif ")
        f.write("-c configs/deconv_rl.yaml\n")

    print(f"\n📄 README created: {readme_path}")
    print("\n" + "=" * 60)
    print("To test deconvolution:")
    print("  python main.py -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \\")
    print("                 -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \\")
    print("                 -c configs/deconv_rl.yaml")


if __name__ == "__main__":
    main()
