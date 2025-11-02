"""
Reson v0 - Main Entry Point
Resolution Enhancement Microscopy Framework

Process images with configurable enhancement pipelines.
Results and metrics are saved per experiment.

Usage:
    # Process directory of images
    python main.py --input data/synthetic/blurred_noisy --config configs/default_v0.yaml
    
    # Process with preset
    python main.py --input data/synthetic/blurred_noisy --config configs/presets/aggressive.yaml
    
    # Process single image
    python main.py --input data/raw/sample.png --config configs/default_v0.yaml
    
    # With ground truth for evaluation
    python main.py --input data/synthetic/blurred_noisy --ground-truth data/synthetic/ground_truth --config configs/default_v0.yaml
"""

from metrics.sharpness import gradient_sharpness, laplacian_variance
from metrics.quality import psnr, ssim, mse
from utils.io import load_image, save_image
from pipeline import load_config, EnhancementPipeline
import sys
import argparse
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def convert_to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj
    sys.path.insert(0, str(project_root))


# Supported image formats
SUPPORTED_FORMATS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def print_progress_bar(current, total, prefix='', suffix='', length=40):
    """Print a progress bar to console."""
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    print(
        f'\r{prefix} |{bar}| [{current}/{total}] {suffix}', end='', flush=True)
    if current == total:
        print()  # New line when complete


def get_image_files(directory):
    """Get all image files from directory."""
    directory = Path(directory)

    if directory.is_file():
        # Single file
        if directory.suffix.lower() in SUPPORTED_FORMATS:
            return [directory]
        else:
            print(f"Error: Unsupported format {directory.suffix}")
            print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
            return []

    # Directory - get all image files
    image_files = []
    for ext in SUPPORTED_FORMATS:
        image_files.extend(directory.glob(f"*{ext}"))
        # Also check uppercase version
        if ext != ext.upper():
            image_files.extend(directory.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    return sorted(list(set(image_files)))


def find_ground_truth(image_path, ground_truth_dir):
    """Find corresponding ground truth image."""
    if ground_truth_dir is None:
        return None

    ground_truth_dir = Path(ground_truth_dir)
    if not ground_truth_dir.exists():
        return None

    # Try to find matching ground truth
    # Remove common suffixes like _degraded, _blurred, _noisy, etc.
    stem = image_path.stem
    for suffix in ['_degraded', '_blurred', '_noisy', '_gaussian', '_poisson', '_saltpepper']:
        stem = stem.replace(suffix, '')

    # Try different extensions
    for ext in SUPPORTED_FORMATS:
        gt_path = ground_truth_dir / f"{stem}{ext}"
        if gt_path.exists():
            return gt_path

    return None


def process_images(input_path, config_path, ground_truth_dir=None, visualize=False, verbose=False):
    """
    Process images from input directory or single file.

    Parameters
    ----------
    input_path : str or Path
        Directory containing images or single image file
    config_path : str or Path
        Path to configuration YAML file
    ground_truth_dir : str or Path, optional
        Directory containing ground truth images for evaluation
    visualize : bool
        Whether to show visualizations
    verbose : bool
        Whether to show detailed processing information

    Returns
    -------
    dict
        Processing results including metrics
    """

    input_path = Path(input_path)
    config_path = Path(config_path)

    # Load configuration
    if not verbose:
        print("="*70)
        print("RESON v0 - Resolution Enhancement Microscopy")
        print("="*70)
    else:
        print("="*70)
        print("RESON v0 - Resolution Enhancement Microscopy")
        print("="*70)
        print(f"\nConfiguration: {config_path.name}")

    config = load_config(str(config_path))
    experiment_name = config.get('experiment_name', config_path.stem)

    if verbose:
        print(f"Experiment: {experiment_name}")
        print("-"*70)

    # Create pipeline
    pipeline = EnhancementPipeline(config, verbose=verbose)

    # Get image files
    image_files = get_image_files(input_path)

    if not image_files:
        print(f"\nError: No images found in {input_path}")
        print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return None

    if not verbose:
        # Minimal output - just show what's being used
        input_display = input_path.name if input_path.is_file() else input_path.name
        print(f"\nDirectory: {input_display}")
        print(
            f"Pipeline:  {' → '.join([m['name'] for m in pipeline.modules])}")
        print(f"Images:    {len(image_files)} files\n")
    else:
        print(f"\nFound {len(image_files)} image(s) to process")
        if ground_truth_dir:
            print(f"Ground truth directory: {ground_truth_dir}")

    # Setup output directories
    output_dir = project_root / "data" / "processed" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = project_root / "results" / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nOutput directory: {output_dir}")
        print(f"Results directory: {results_dir}")
        print("="*70)

    # Process each image
    all_results = []
    start_time = time.time()

    for idx, img_path in enumerate(image_files, 1):
        if verbose:
            print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
            print("-"*70)
        else:
            # Show progress bar
            print_progress_bar(idx - 1, len(image_files),
                               prefix='Progress:', suffix='')

        try:
            # Load image
            img = load_image(img_path, as_float=True)

            if verbose:
                print(f"  Loaded: {img.shape} | dtype: {img.dtype}")
                print(f"  Intensity range: [{img.min():.4f}, {img.max():.4f}]")

            # Process
            enhanced = pipeline.process(
                img, compute_metrics=True, visualize=False, verbose=verbose)

            # Save enhanced image
            output_path = output_dir / \
                f"{img_path.stem}_enhanced{img_path.suffix}"
            save_image(enhanced, output_path, bit_depth=16)

            if verbose:
                print(f"  Saved: {output_path.name}")

            # Compute metrics
            result = {
                'image': img_path.name,
                'output': output_path.name,
                'shape': list(img.shape),
                'sharpness_after': gradient_sharpness(enhanced),
                'laplacian_after': laplacian_variance(enhanced),
            }

            # Get pipeline metrics
            pipeline_metrics = pipeline.get_metrics()
            result.update(pipeline_metrics)

            # Evaluate against ground truth if available
            gt_path = find_ground_truth(img_path, ground_truth_dir)
            if gt_path and verbose:
                print(f"  Ground truth: {gt_path.name}")
                gt_img = load_image(gt_path, as_float=True)

                # Compute quality metrics
                result['psnr_before'] = psnr(gt_img, img)
                result['psnr_after'] = psnr(gt_img, enhanced)
                result['psnr_improvement'] = result['psnr_after'] - \
                    result['psnr_before']

                result['ssim_before'] = ssim(gt_img, img)
                result['ssim_after'] = ssim(gt_img, enhanced)
                result['ssim_improvement'] = result['ssim_after'] - \
                    result['ssim_before']

                result['mse_before'] = mse(gt_img, img)
                result['mse_after'] = mse(gt_img, enhanced)

                result['sharpness_before'] = gradient_sharpness(img)

                # Print evaluation
                print(f"\n  Evaluation vs Ground Truth:")
                print(
                    f"    PSNR:      {result['psnr_before']:6.2f} → {result['psnr_after']:6.2f} dB  (Δ {result['psnr_improvement']:+.2f})")
                print(
                    f"    SSIM:      {result['ssim_before']:6.4f} → {result['ssim_after']:6.4f}  (Δ {result['ssim_improvement']:+.4f})")
                print(
                    f"    Sharpness: {result['sharpness_before']:6.4f} → {result['sharpness_after']:6.4f}")
            elif gt_path and not verbose:
                # Still compute metrics, just don't print
                gt_img = load_image(gt_path, as_float=True)
                result['psnr_before'] = psnr(gt_img, img)
                result['psnr_after'] = psnr(gt_img, enhanced)
                result['psnr_improvement'] = result['psnr_after'] - \
                    result['psnr_before']
                result['ssim_before'] = ssim(gt_img, img)
                result['ssim_after'] = ssim(gt_img, enhanced)
                result['ssim_improvement'] = result['ssim_after'] - \
                    result['ssim_before']
                result['mse_before'] = mse(gt_img, img)
                result['mse_after'] = mse(gt_img, enhanced)
                result['sharpness_before'] = gradient_sharpness(img)
            elif verbose:
                print(f"\n  Quality Metrics:")
                print(f"    Sharpness:  {result['sharpness_after']:.4f}")
                print(f"    Laplacian:  {result['laplacian_after']:.4f}")

            # Save individual result
            result_file = results_dir / f"{img_path.stem}_result.json"
            with open(result_file, 'w') as f:
                json.dump(convert_to_python_types(result), f, indent=2)

            all_results.append(result)

        except Exception as e:
            if verbose:
                print(f"  Error processing {img_path.name}: {e}")
            else:
                # Update progress bar even on error
                print_progress_bar(idx, len(image_files),
                                   prefix='Progress:', suffix='ERROR')
            continue

    # Final progress bar update
    if not verbose:
        print_progress_bar(len(image_files), len(image_files),
                           prefix='Progress:', suffix='Complete')
        elapsed = time.time() - start_time
        print(f"Time elapsed: {elapsed:.1f}s\n")

    # Generate overall report
    if all_results:
        if not verbose:
            print("\n" + "="*70)
            print("✓ Processing complete!")
            print("="*70)
            print(f"Enhanced images: {output_dir}")
            print(f"Results saved:   {results_dir}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("OVERALL RESULTS")
            print("="*70)

        # Compute averages
        report = {
            'experiment_name': experiment_name,
            'config_file': str(config_path),
            'timestamp': datetime.now().isoformat(),
            'total_images': len(all_results),
            'output_directory': str(output_dir),
        }

        # Average metrics
        if 'psnr_improvement' in all_results[0]:
            # Has ground truth evaluation
            report['avg_psnr_before'] = np.mean(
                [r['psnr_before'] for r in all_results])
            report['avg_psnr_after'] = np.mean(
                [r['psnr_after'] for r in all_results])
            report['avg_psnr_improvement'] = np.mean(
                [r['psnr_improvement'] for r in all_results])

            report['avg_ssim_before'] = np.mean(
                [r['ssim_before'] for r in all_results])
            report['avg_ssim_after'] = np.mean(
                [r['ssim_after'] for r in all_results])
            report['avg_ssim_improvement'] = np.mean(
                [r['ssim_improvement'] for r in all_results])

            report['avg_sharpness_before'] = np.mean(
                [r['sharpness_before'] for r in all_results])
            report['avg_sharpness_after'] = np.mean(
                [r['sharpness_after'] for r in all_results])

            if verbose:
                print(f"\nWith Ground Truth Evaluation:")
                print(
                    f"  Average PSNR Improvement:  {report['avg_psnr_improvement']:+.2f} dB")
                print(
                    f"  Average SSIM Improvement:  {report['avg_ssim_improvement']:+.4f}")
                print(
                    f"  Final Average PSNR:        {report['avg_psnr_after']:.2f} dB")
                print(
                    f"  Final Average SSIM:        {report['avg_ssim_after']:.4f}")
        else:
            # No ground truth
            report['avg_sharpness'] = np.mean(
                [r['sharpness_after'] for r in all_results])
            report['avg_laplacian'] = np.mean(
                [r['laplacian_after'] for r in all_results])

            if verbose:
                print(f"\nQuality Metrics:")
                print(f"  Average Sharpness:  {report['avg_sharpness']:.4f}")
                print(f"  Average Laplacian:  {report['avg_laplacian']:.4f}")

        # Save overall report
        report['individual_results'] = all_results
        report_file = results_dir / "overall_report.json"
        with open(report_file, 'w') as f:
            json.dump(convert_to_python_types(report), f, indent=2)

        if verbose:
            print(f"\n✓ Overall report saved: {report_file}")
            print(f"✓ Enhanced images saved: {output_dir}")
            print("="*70)

        return report

    return None


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Reson v0 - Resolution Enhancement for Microscopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory of images
  python main.py -i data/synthetic/blurred_noisy -c configs/default_v0.yaml
  
  # Process with ground truth evaluation
  python main.py -i data/synthetic/blurred_noisy -g data/synthetic/ground_truth -c configs/presets/aggressive.yaml
  
  # Process single image
  python main.py -i data/raw/sample.png -c configs/default_v0.yaml
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input directory or single image file (supports PNG, JPG, TIF)')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Configuration file path')
    parser.add_argument('-g', '--ground-truth', type=str, default=None,
                        help='Ground truth directory for evaluation (optional)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualizations during processing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed processing information')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input path not found: {args.input}")
        return

    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return

    # Process images
    process_images(
        input_path=args.input,
        config_path=args.config,
        ground_truth_dir=args.ground_truth,
        visualize=args.visualize,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
