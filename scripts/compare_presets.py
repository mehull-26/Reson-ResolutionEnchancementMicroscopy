"""
Compare performance of different enhancement presets.
"""
import json
from pathlib import Path


def compare_presets():
    """Compare all preset results."""
    results_dir = Path("results")

    if not results_dir.exists():
        print("No results found. Run main.py first.")
        return

    # Load all reports
    reports = {}
    for report_file in results_dir.glob("*/overall_report.json"):
        experiment_name = report_file.parent.name
        with open(report_file, 'r') as f:
            reports[experiment_name] = json.load(f)

    if not reports:
        print("No reports found.")
        return

    print("\n" + "="*80)
    print("PRESET COMPARISON - Resolution Enhancement Microscopy")
    print("="*80)

    # Build comparison table
    print(f"\n{'Preset':<15} {'PSNR Δ (dB)':>12} {'SSIM Δ':>12} {'Final PSNR':>14} {'Final SSIM':>14}")
    print("-"*80)

    for name in sorted(reports.keys()):
        r = reports[name]
        if 'avg_psnr_improvement' in r:
            print(f"{name:<15} {r['avg_psnr_improvement']:>11.2f} {r['avg_ssim_improvement']:>12.4f} "
                  f"{r['avg_psnr_after']:>13.2f} dB {r['avg_ssim_after']:>13.4f}")

    # Show best performers
    if reports:
        print("\n" + "="*80)
        print("BEST PERFORMERS")
        print("="*80)

        # Best PSNR improvement
        best_psnr = max(reports.items(),
                        key=lambda x: x[1].get('avg_psnr_improvement', -float('inf')))
        print(
            f"Best PSNR Improvement: {best_psnr[0]} ({best_psnr[1]['avg_psnr_improvement']:+.2f} dB)")

        # Best SSIM improvement
        best_ssim = max(reports.items(),
                        key=lambda x: x[1].get('avg_ssim_improvement', -float('inf')))
        print(
            f"Best SSIM Improvement: {best_ssim[0]} ({best_ssim[1]['avg_ssim_improvement']:+.4f})")

        # Best final PSNR
        best_final_psnr = max(reports.items(),
                              key=lambda x: x[1].get('avg_psnr_after', -float('inf')))
        print(
            f"Best Final PSNR:       {best_final_psnr[0]} ({best_final_psnr[1]['avg_psnr_after']:.2f} dB)")

        # Best final SSIM
        best_final_ssim = max(reports.items(),
                              key=lambda x: x[1].get('avg_ssim_after', -float('inf')))
        print(
            f"Best Final SSIM:       {best_final_ssim[0]} ({best_final_ssim[1]['avg_ssim_after']:.4f})")

        print("="*80)


if __name__ == "__main__":
    compare_presets()
