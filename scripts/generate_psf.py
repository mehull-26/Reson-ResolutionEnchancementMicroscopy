"""
PSF Generation and Visualization Utility

Interactive tool to generate, visualize, and save PSFs for deconvolution.

Usage:
  python scripts/generate_psf.py --method gaussian --wavelength 550 --na 1.4
  python scripts/generate_psf.py --method gibson_lanni --wavelength 520 --na 1.4
  python scripts/generate_psf.py --load measured_psf.tif --visualize

Author: Mehul Patel
Date: November 2025
"""

from utils.io import save_image
from utils.psf_generation import (
    generate_gaussian_psf, generate_airy_psf, generate_gibson_lanni_psf,
    load_custom_psf, visualize_psf, get_psf
)
import argparse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate and visualize PSFs for deconvolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # PSF method
    parser.add_argument('--method', '-m', type=str, default='gaussian',
                        choices=['gaussian', 'airy', 'gibson_lanni', 'load'],
                        help='PSF generation method')

    # Common parameters
    parser.add_argument('--wavelength', '-w', type=float, default=550,
                        help='Wavelength in nanometers')
    parser.add_argument('--na', type=float, default=1.4,
                        help='Numerical aperture')
    parser.add_argument('--pixel-size', '-p', type=float, default=0.065,
                        help='Pixel size in micrometers')
    parser.add_argument('--size', '-s', type=int, default=31,
                        help='PSF size (odd number)')

    # Gibson-Lanni specific
    parser.add_argument('--ni', type=float, default=1.518,
                        help='Refractive index of immersion (Gibson-Lanni)')
    parser.add_argument('--ns', type=float, default=1.33,
                        help='Refractive index of sample (Gibson-Lanni)')
    parser.add_argument('--ti', type=float, default=150,
                        help='Working distance in µm (Gibson-Lanni)')

    # Load custom PSF
    parser.add_argument('--load', '-l', type=str, default=None,
                        help='Load PSF from file')

    # Output options
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename (TIF, NPY, or NPZ)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Show visualization')
    parser.add_argument('--no-save', action='store_true',
                        help='Don\'t save PSF to file')

    # Presets
    parser.add_argument('--preset', type=str, default=None,
                        choices=['fluorescence_oil', 'fluorescence_water',
                                 'brightfield_dry', 'brightfield_oil'],
                        help='Use preset configuration')

    return parser.parse_args()


def get_preset_params(preset):
    """Get parameters for common microscopy configurations."""
    presets = {
        'fluorescence_oil': {
            'method': 'gibson_lanni',
            'wavelength': 520,  # GFP
            'na': 1.4,
            'pixel_size': 0.065,
            'size': 41,
            'ni': 1.518,
            'ns': 1.33,
            'ti': 150
        },
        'fluorescence_water': {
            'method': 'gibson_lanni',
            'wavelength': 520,
            'na': 1.2,
            'pixel_size': 0.065,
            'size': 41,
            'ni': 1.33,
            'ns': 1.33,
            'ti': 150
        },
        'brightfield_dry': {
            'method': 'gaussian',
            'wavelength': 550,
            'na': 0.75,
            'pixel_size': 0.1,
            'size': 31
        },
        'brightfield_oil': {
            'method': 'airy',
            'wavelength': 550,
            'na': 1.4,
            'pixel_size': 0.065,
            'size': 31
        }
    }
    return presets.get(preset)


def main():
    """Main function."""
    args = parse_args()

    print("PSF Generation Utility")
    print("=" * 60)

    # Use preset if specified
    if args.preset:
        print(f"Using preset: {args.preset}")
        preset_params = get_preset_params(args.preset)
        for key, value in preset_params.items():
            setattr(args, key, value)

    # Generate or load PSF
    if args.load:
        print(f"\nLoading PSF from: {args.load}")
        psf = load_custom_psf(args.load)
        psf_name = Path(args.load).stem
    elif args.method == 'load':
        print("Error: --load filename required for 'load' method")
        return
    else:
        print(f"\nGenerating {args.method} PSF...")
        print(f"  Wavelength: {args.wavelength} nm")
        print(f"  Numerical Aperture: {args.na}")
        print(f"  Pixel Size: {args.pixel_size} µm")
        print(f"  PSF Size: {args.size} x {args.size}")

        if args.method == 'gaussian':
            psf = generate_gaussian_psf(
                wavelength=args.wavelength,
                numerical_aperture=args.na,
                pixel_size=args.pixel_size,
                size=args.size
            )
            psf_name = f"gaussian_w{args.wavelength}_na{args.na}"

        elif args.method == 'airy':
            psf = generate_airy_psf(
                wavelength=args.wavelength,
                numerical_aperture=args.na,
                pixel_size=args.pixel_size,
                size=args.size
            )
            psf_name = f"airy_w{args.wavelength}_na{args.na}"

        elif args.method == 'gibson_lanni':
            print(f"  Immersion RI (ni): {args.ni}")
            print(f"  Sample RI (ns): {args.ns}")
            print(f"  Working Distance: {args.ti} µm")
            psf = generate_gibson_lanni_psf(
                wavelength=args.wavelength,
                numerical_aperture=args.na,
                pixel_size=args.pixel_size,
                size=args.size,
                ni=args.ni,
                ns=args.ns,
                ti=args.ti
            )
            psf_name = f"gibson_lanni_w{args.wavelength}_na{args.na}"

    # Print PSF statistics
    print(f"\nPSF Statistics:")
    print(f"  Shape: {psf.shape}")
    print(f"  Sum: {psf.sum():.6f} (should be ~1.0)")
    print(f"  Max: {psf.max():.6f}")
    print(f"  Min: {psf.min():.6f}")
    print(f"  Center value: {psf[psf.shape[0]//2, psf.shape[1]//2]:.6f}")

    # Calculate FWHM (Full Width at Half Maximum)
    center = psf.shape[0] // 2
    center_row = psf[center, :]
    half_max = center_row.max() / 2
    above_half = center_row > half_max
    fwhm = np.sum(above_half) * \
        args.pixel_size if args.method != 'load' else np.sum(above_half)
    print(
        f"  Estimated FWHM: {fwhm:.3f} {'µm' if args.method != 'load' else 'pixels'}")

    # Save PSF
    if not args.no_save:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"{psf_name}.tif")

        # Determine format
        ext = output_path.suffix.lower()
        if ext in ['.npy']:
            np.save(output_path, psf)
        elif ext in ['.npz']:
            np.savez(output_path, psf=psf)
        elif ext in ['.tif', '.tiff', '.png']:
            save_image(psf, output_path, bit_depth=16)
        else:
            # Default to TIF
            output_path = output_path.with_suffix('.tif')
            save_image(psf, output_path, bit_depth=16)

        print(f"\n✅ PSF saved to: {output_path}")

        # Also save as NPY for easy loading
        if ext not in ['.npy', '.npz']:
            npy_path = output_path.with_suffix('.npy')
            np.save(npy_path, psf)
            print(f"✅ PSF also saved as: {npy_path}")

    # Visualize
    if args.visualize:
        print("\n📊 Displaying PSF visualization...")
        try:
            visualize_psf(psf, title=f"{psf_name}")
        except ImportError:
            print(
                "❌ Visualization requires matplotlib. Install with: pip install matplotlib")
        except Exception as e:
            print(f"❌ Visualization error: {e}")

    print("\n" + "=" * 60)
    print("To use this PSF for deconvolution:")
    print(f"  1. Update config YAML: psf_method: 'custom'")
    print(
        f"  2. Set psf_params: {{psf_file: '{output_path if not args.no_save else 'your_psf.tif'}'}}")
    print(
        f"  3. Or use in code: psf = load_custom_psf('{output_path if not args.no_save else 'your_psf.tif'}')")


if __name__ == "__main__":
    main()
