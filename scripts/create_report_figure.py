"""
Generate Figure 1 for the report showing deconvolution results.
Creates a 3-row visualization with heat map colormaps:
- Row 1: Checkerboard clean (Gaussian PSF)
- Row 2: Checkerboard noisy (RL-TV)
- Row 3: Fluorescent beads (Gibson-Lanni)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def load_tif(path):
    """Load TIFF image and normalize to [0, 1]."""
    img = np.array(Image.open(path))
    return img.astype(float) / img.max()


def create_figure():
    """Create the 3x3 comparison figure."""

    # Define paths
    data_dir = Path('data/synthetic_psf')
    processed_dir = Path('data/processed')

    # Load images for each row
    # Row 1: Checkerboard clean (Gaussian)
    row1_gt = load_tif(
        data_dir / 'checkerboard_gaussian_clean/ground_truth.tif')
    row1_deg = load_tif(data_dir / 'checkerboard_gaussian_clean/degraded.tif')
    row1_enh = load_tif(
        processed_dir / 'deconv_rl_known/degraded_enhanced.tif')

    # Row 2: Checkerboard noisy (RL-TV)
    row2_gt = load_tif(
        data_dir / 'checkerboard_gaussian_poisson3/ground_truth.tif')
    row2_deg = load_tif(
        data_dir / 'checkerboard_gaussian_poisson3/degraded.tif')
    row2_enh = load_tif(
        processed_dir / 'rl_noisy_optimized/degraded_enhanced.tif')

    # Row 3: Fluorescent beads (Gibson-Lanni)
    row3_gt = load_tif(data_dir / 'beads_gibson_lanni_clean/ground_truth.tif')
    row3_deg = load_tif(data_dir / 'beads_gibson_lanni_clean/degraded.tif')
    row3_enh = load_tif(
        processed_dir / 'deconv_rl_gibson_lanni/degraded_enhanced.tif')

    # Create figure with 3 rows, 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Row 1: Checkerboard Clean (use gray colormap for clean visualization)
    axes[0, 0].imshow(row1_gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('(a) Ground Truth', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(row1_deg, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('(b) Degraded (Gaussian PSF)',
                         fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(row1_enh, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('(c) Restored (+5.12 dB)',
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Add row label
    axes[0, 0].text(-0.15, 0.5, 'Checkerboard\nClean', transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

    # Row 2: Checkerboard Noisy (use gray colormap for clean visualization)
    axes[1, 0].imshow(row2_gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('(d) Ground Truth', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(row2_deg, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('(e) Degraded (Poisson SNR≈3)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(row2_enh, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('(f) RL-TV Restored (+3.36 dB)',
                         fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')

    axes[1, 0].text(-0.15, 0.5, 'Checkerboard\nNoisy', transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

    # Row 3: Fluorescent Beads (use grayscale like the rest)
    axes[2, 0].imshow(row3_gt, cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title('(g) Synthetic Beads', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')

    axes[2, 1].imshow(row3_deg, cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title('(h) Degraded (Gibson-Lanni)',
                         fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')

    axes[2, 2].imshow(row3_enh, cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title('(i) Restored (+3.84 dB)',
                         fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')

    axes[2, 0].text(-0.15, 0.5, 'Fluorescent\nBeads', transform=axes[2, 0].transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right', rotation=90)

    # Adjust spacing
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.02,
                        wspace=0.05, hspace=0.12)

    # Save figure
    output_path = Path('report/figure1.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figure saved to: {output_path}")

    plt.close()


if __name__ == '__main__':
    create_figure()
