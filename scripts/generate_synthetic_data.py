"""
Generate synthetic microscopy images for testing enhancement algorithms.

Creates ground truth images, then applies:
- Gaussian blur (simulating diffraction)
- Noise (Gaussian, Poisson, salt-pepper)
- Combined degradations

This allows quantitative evaluation with known ground truth.
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy import signal


# Get project root
project_root = Path(__file__).parent.parent


def save_image_simple(img, filepath, bit_depth=16):
    """Simple image saving function."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Clip and convert
    img = np.clip(img, 0, 1)

    if bit_depth == 8:
        img_out = (img * 255).astype(np.uint8)
    else:
        img_out = (img * 65535).astype(np.uint16)

    # Save
    if len(img_out.shape) == 3:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(filepath), img_out)


def create_circles_pattern(size=(512, 512), n_circles=20):
    """Create synthetic image with circles of various sizes."""
    img = np.zeros(size, dtype=np.float32)

    for _ in range(n_circles):
        center = (np.random.randint(
            50, size[1]-50), np.random.randint(50, size[0]-50))
        radius = np.random.randint(10, 40)
        intensity = np.random.uniform(0.5, 1.0)
        cv2.circle(img, center, radius, intensity, -1)

    return img


def create_cells_pattern(size=(512, 512), n_cells=15):
    """Create synthetic cell-like structures."""
    img = np.zeros(size, dtype=np.float32)

    for _ in range(n_cells):
        center = (np.random.randint(
            100, size[1]-100), np.random.randint(100, size[0]-100))

        # Cell body (ellipse)
        axes = (np.random.randint(20, 50), np.random.randint(15, 40))
        angle = np.random.randint(0, 180)
        cv2.ellipse(img, center, axes, angle, 0, 360, 0.7, -1)

        # Nucleus (smaller circle)
        nucleus_offset = (np.random.randint(-10, 10),
                          np.random.randint(-10, 10))
        nucleus_center = (center[0] + nucleus_offset[0],
                          center[1] + nucleus_offset[1])
        nucleus_radius = np.random.randint(8, 15)
        cv2.circle(img, nucleus_center, nucleus_radius, 0.9, -1)

    return img


def create_lines_pattern(size=(512, 512), n_lines=30):
    """Create lines pattern (simulating fibers/structures)."""
    img = np.zeros(size, dtype=np.float32)

    for _ in range(n_lines):
        pt1 = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
        pt2 = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
        thickness = np.random.randint(1, 4)
        intensity = np.random.uniform(0.6, 1.0)
        cv2.line(img, pt1, pt2, intensity, thickness)

    return img


def create_grid_pattern(size=(512, 512), spacing=50, line_width=3):
    """Create grid pattern (simulating structured samples)."""
    img = np.zeros(size, dtype=np.float32)

    # Vertical lines
    for x in range(0, size[1], spacing):
        img[:, max(0, x-line_width//2):min(size[1], x+line_width//2)] = 0.8

    # Horizontal lines
    for y in range(0, size[0], spacing):
        img[max(0, y-line_width//2):min(size[0], y+line_width//2), :] = 0.8

    return img


def create_mixed_pattern(size=(512, 512)):
    """Create complex pattern with multiple features."""
    img = np.zeros(size, dtype=np.float32)

    # Background texture
    noise = np.random.uniform(0, 0.1, size)
    img += noise

    # Add cells
    cells = create_cells_pattern(size, n_cells=10)
    img = np.maximum(img, cells)

    # Add some fibers
    lines = create_lines_pattern(size, n_lines=15) * 0.6
    img = np.maximum(img, lines)

    # Add some spots
    circles = create_circles_pattern(size, n_circles=10) * 0.5
    img = np.maximum(img, circles)

    return np.clip(img, 0, 1)


def create_siemens_star(size=(512, 512), n_spokes=36):
    """Create Siemens star for resolution testing."""
    img = np.zeros(size, dtype=np.float32)
    center = (size[1] // 2, size[0] // 2)
    max_radius = min(size) // 2 - 10

    for i in range(n_spokes):
        angle1 = 2 * np.pi * i / n_spokes
        angle2 = 2 * np.pi * (i + 0.5) / n_spokes

        # Create filled triangle (spoke)
        pts = np.array([
            center,
            (int(center[0] + max_radius * np.cos(angle1)),
             int(center[1] + max_radius * np.sin(angle1))),
            (int(center[0] + max_radius * np.cos(angle2)),
             int(center[1] + max_radius * np.sin(angle2)))
        ], dtype=np.int32)

        if i % 2 == 0:
            cv2.fillPoly(img, [pts], 1.0)

    return img


def apply_gaussian_blur(img, sigma=2.0):
    """Simulate diffraction-limited blur."""
    return gaussian_filter(img, sigma=sigma)


def apply_psf_blur(img, psf_size=15):
    """Apply Point Spread Function blur (Gaussian PSF)."""
    # Create Gaussian PSF
    x = np.linspace(-3, 3, psf_size)
    y = np.linspace(-3, 3, psf_size)
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X**2 + Y**2) / 2)
    psf = psf / psf.sum()

    # Convolve with PSF
    blurred = signal.convolve2d(img, psf, mode='same', boundary='symm')
    return np.clip(blurred, 0, 1)


def add_gaussian_noise(img, sigma=0.05):
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)


def add_poisson_noise(img, scale=100):
    """Add Poisson noise (photon counting noise)."""
    # Scale up, add Poisson noise, scale back
    scaled = img * scale
    noisy = np.random.poisson(scaled).astype(np.float32)
    noisy = noisy / scale
    return np.clip(noisy, 0, 1)


def add_salt_pepper_noise(img, amount=0.01):
    """Add salt and pepper noise."""
    noisy = img.copy()

    # Salt (white pixels)
    n_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i, n_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 1.0

    # Pepper (black pixels)
    n_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i, n_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0.0

    return noisy


def generate_dataset(output_dir, size=(512, 512)):
    """Generate complete synthetic dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    (output_dir / "blurred").mkdir(exist_ok=True)
    (output_dir / "noisy").mkdir(exist_ok=True)
    (output_dir / "blurred_noisy").mkdir(exist_ok=True)

    patterns = {
        'circles': create_circles_pattern(size),
        'cells': create_cells_pattern(size),
        'lines': create_lines_pattern(size),
        'grid': create_grid_pattern(size),
        'mixed': create_mixed_pattern(size),
        'siemens_star': create_siemens_star(size),
    }

    print("Generating synthetic dataset...")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {size}")
    print()

    for name, img in patterns.items():
        print(f"Processing: {name}")

        # Save ground truth
        save_image_simple(img, output_dir / "ground_truth" /
                          f"{name}.tif", bit_depth=16)

        # Apply blur only
        blurred = apply_psf_blur(img, psf_size=15)
        save_image_simple(blurred, output_dir / "blurred" /
                          f"{name}_blurred.tif", bit_depth=16)

        # Apply noise only (multiple types)
        noisy_gaussian = add_gaussian_noise(img, sigma=0.05)
        save_image_simple(noisy_gaussian, output_dir / "noisy" /
                          f"{name}_gaussian.tif", bit_depth=16)

        noisy_poisson = add_poisson_noise(img, scale=100)
        save_image_simple(noisy_poisson, output_dir / "noisy" /
                          f"{name}_poisson.tif", bit_depth=16)

        noisy_sp = add_salt_pepper_noise(img, amount=0.01)
        save_image_simple(noisy_sp, output_dir / "noisy" /
                          f"{name}_saltpepper.tif", bit_depth=16)

        # Apply blur + noise (realistic microscopy scenario)
        blurred_noisy = apply_psf_blur(img, psf_size=15)
        blurred_noisy = add_gaussian_noise(blurred_noisy, sigma=0.03)
        save_image_simple(blurred_noisy, output_dir / "blurred_noisy" /
                          f"{name}_degraded.tif", bit_depth=16)

        print(f"  ✓ Ground truth: {name}.tif")
        print(f"  ✓ Blurred: {name}_blurred.tif")
        print(
            f"  ✓ Noisy: {name}_gaussian.tif, {name}_poisson.tif, {name}_saltpepper.tif")
        print(f"  ✓ Blurred+Noisy: {name}_degraded.tif")
        print()

    # Create metadata file
    metadata = f"""# Synthetic Dataset Metadata

Generated: {size[0]}x{size[1]} images

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
"""

    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(metadata)

    print("=" * 60)
    print("Dataset generation complete!")
    print(f"Total images: {len(patterns) * 7}")
    print(f"Location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    output_dir = project_root / "data" / "synthetic"
    generate_dataset(output_dir, size=(512, 512))
