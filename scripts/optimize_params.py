"""
Test different iteration and regularization combinations to find optimal visual results.
"""

import json
import subprocess
import yaml
from pathlib import Path
import shutil


def test_config(iterations, regularization, name):
    """Test a specific configuration."""
    config_path = Path('configs/deconv_rl_known.yaml')

    # Read config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Modify parameters
    config['enhancement']['modules'][0]['params']['iterations'] = iterations
    config['enhancement']['modules'][0]['params']['regularization'] = regularization

    # Write temporary config
    temp_config = Path('configs/_temp_test.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Run test
    cmd = [
        'python', 'main.py',
        '-i', 'data/synthetic_psf/checkerboard_gaussian_clean/degraded.tif',
        '-g', 'data/synthetic_psf/checkerboard_gaussian_clean/ground_truth.tif',
        '-c', str(temp_config)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, env={
                            'PYTHONIOENCODING': 'utf-8'})

    # Read results
    result_file = Path('results/_temp_test/degraded_result.json')
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        return {
            'name': name,
            'iterations': iterations,
            'regularization': regularization,
            'psnr_before': data['psnr_before'],
            'psnr_after': data['psnr_after'],
            'psnr_improvement': data['psnr_improvement'],
            'ssim': data['ssim_after']
        }
    return None


def main():
    """Test various parameter combinations."""

    # Test configurations
    tests = [
        # Current baseline
        (50, 0.001, "Baseline (current)"),

        # More iterations
        (100, 0.001, "More iterations"),
        (150, 0.001, "Even more iterations"),

        # Less regularization (sharper but maybe noisy)
        (50, 0.0001, "Less regularization"),
        (100, 0.0001, "100 iter + less reg"),

        # No regularization
        (50, 0.0, "No regularization"),
        (100, 0.0, "100 iter + no reg"),

        # More regularization (smoother)
        (50, 0.005, "More regularization"),
        (100, 0.005, "100 iter + more reg"),
    ]

    print("=" * 80)
    print("TESTING DIFFERENT PARAMETER COMBINATIONS")
    print("=" * 80)
    print()

    results = []
    for iterations, reg, name in tests:
        print(
            f"Testing: {name:25s} (iter={iterations:3d}, λ={reg:.4f})... ", end='', flush=True)
        result = test_config(iterations, reg, name)
        if result:
            results.append(result)
            print(
                f"✓ PSNR: {result['psnr_before']:.2f} → {result['psnr_after']:.2f} dB (+{result['psnr_improvement']:.2f} dB), SSIM={result['ssim']:.4f}")
        else:
            print("✗ Failed")

    print()
    print("=" * 80)
    print("SUMMARY - SORTED BY PSNR IMPROVEMENT")
    print("=" * 80)
    print()

    results.sort(key=lambda x: x['psnr_improvement'], reverse=True)

    print(f"{'Configuration':<30s} {'Iter':<6s} {'Lambda':<10s} {'Before':<10s} {'After':<10s} {'Δ PSNR':<10s} {'SSIM':<8s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30s} {r['iterations']:<6d} {r['regularization']:<10.4f} "
              f"{r['psnr_before']:<10.2f} {r['psnr_after']:<10.2f} "
              f"+{r['psnr_improvement']:<9.2f} {r['ssim']:<8.4f}")

    print()
    print("=" * 80)
    print(f"BEST RESULT: {results[0]['name']}")
    print(f"  Iterations: {results[0]['iterations']}")
    print(f"  Regularization: {results[0]['regularization']}")
    print(f"  PSNR Improvement: +{results[0]['psnr_improvement']:.2f} dB")
    print(f"  SSIM: {results[0]['ssim']:.4f}")
    print("=" * 80)

    # Cleanup
    temp_config = Path('configs/_temp_test.yaml')
    if temp_config.exists():
        temp_config.unlink()


if __name__ == '__main__':
    main()
