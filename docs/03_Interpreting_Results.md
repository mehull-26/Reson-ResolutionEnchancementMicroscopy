# Interpreting Results Guide

## Reson v0 - Understanding Metrics and Output Files

---

## Table of Contents
1. [Output Files Overview](#output-files-overview)
2. [Understanding Metrics](#understanding-metrics)
3. [Reading JSON Reports](#reading-json-reports)
4. [Quality Assessment](#quality-assessment)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting Results](#troubleshooting-results)

---

## Output Files Overview

After processing, Reson generates two types of output:

### 1. Enhanced Images
**Location:** `data/processed/[experiment_name]/`

```
data/processed/
└── default_v0/
    ├── Z9_0_enhanced.jpg
    ├── Z9_1_enhanced.jpg
    └── ...
```

- **Format:** Same as input (JPG/PNG/TIF)
- **Naming:** `[original_name]_enhanced.[ext]`
- **Bit depth:** 8-bit for JPG, 16-bit for TIF/PNG (configurable)

### 2. Metric Reports
**Location:** `results/[experiment_name]/`

```
results/
└── default_v0/
    ├── Z9_0_result.json       # Per-image metrics
    ├── Z9_1_result.json
    ├── ...
    └── overall_report.json    # Summary statistics
```

---

## Understanding Metrics

Reson computes two categories of metrics:

### 1. Sharpness Metrics (Always Computed)

#### **Gradient Sharpness**
- **Range:** 0.0 to 1.0 (unbounded, but typically < 0.5)
- **Meaning:** Measures edge strength using image gradients
- **Higher is better:** More defined edges

**Interpretation:**
- `< 0.05`: Very blurry/soft
- `0.05 - 0.15`: Moderate sharpness (typical microscopy)
- `0.15 - 0.30`: Sharp
- `> 0.30`: Very sharp (or high-frequency noise)

**Use case:** Quick assessment of overall sharpness

#### **Laplacian Variance**
- **Range:** 0 to ∞ (typically 0-1000 for microscopy)
- **Meaning:** Measures focus/sharpness using second derivatives
- **Higher is better:** Better focus

**Interpretation:**
- `< 50`: Out of focus
- `50 - 200`: Acceptable focus
- `200 - 500`: Good focus
- `> 500`: Excellent focus

**Use case:** Detecting out-of-focus images

### 2. Quality Metrics (With Ground Truth)

Only available when processing with `--ground-truth` flag:

#### **PSNR (Peak Signal-to-Noise Ratio)**
- **Range:** 0 to ∞ dB (typically 15-40 dB)
- **Meaning:** Measures pixel-wise accuracy vs ground truth
- **Higher is better:** Closer to perfect quality

**Interpretation:**
- `< 20 dB`: Poor quality
- `20-25 dB`: Fair quality
- `25-30 dB`: Good quality
- `30-40 dB`: Excellent quality
- `> 40 dB`: Near-perfect

**Use case:** Quantifying improvement over degraded input

#### **SSIM (Structural Similarity Index)**
- **Range:** -1 to 1 (typically 0.5-1.0)
- **Meaning:** Measures structural similarity vs ground truth
- **Higher is better:** Better perceptual quality

**Interpretation:**
- `< 0.50`: Poor similarity
- `0.50-0.70`: Fair similarity
- `0.70-0.90`: Good similarity
- `0.90-0.99`: Excellent similarity
- `> 0.99`: Near-identical

**Use case:** Perceptual quality assessment

#### **MSE (Mean Squared Error)**
- **Range:** 0 to ∞ (lower is better)
- **Meaning:** Average pixel-wise error squared
- **Lower is better:** Smaller error

**Interpretation:**
- `< 0.001`: Excellent
- `0.001-0.01`: Good
- `0.01-0.05`: Fair
- `> 0.05`: Poor

**Use case:** Technical error measurement

---

## Reading JSON Reports

### Per-Image Result File

**Example:** `results/default_v0/Z9_0_result.json`

```json
{
  "image": "Z9_0.jpg",
  "output": "Z9_0_enhanced.jpg",
  "shape": [1024, 1024],
  "sharpness_after": 0.0537,
  "laplacian_after": 2377.0739,
  "sharpness": 0.0537,
  "laplacian_variance": 2377.0739
}
```

**Key fields:**
- `image`: Input filename
- `output`: Enhanced filename
- `shape`: Image dimensions [height, width]
- `sharpness_after`: Gradient sharpness of enhanced image
- `laplacian_after`: Laplacian variance of enhanced image

### Overall Report (Summary)

**Example:** `results/default_v0/overall_report.json`

```json
{
  "experiment_name": "default_v0",
  "config_file": "configs\\default_v0.yaml",
  "timestamp": "2025-11-02T15:30:45.123456",
  "total_images": 100,
  "output_directory": "B:\\...\\data\\processed\\default_v0",
  "avg_sharpness": 0.0421,
  "avg_laplacian": 156.82,
  "individual_results": [...]
}
```

**Key fields:**
- `experiment_name`: Preset used
- `timestamp`: Processing time
- `total_images`: Number processed
- `avg_sharpness`: Mean sharpness across all images
- `avg_laplacian`: Mean laplacian variance
- `individual_results`: Array of all per-image results

### With Ground Truth Evaluation

When processing synthetic data with ground truth:

```json
{
  "image": "cells_degraded.tif",
  "output": "cells_degraded_enhanced.tif",
  "psnr_before": 25.85,
  "psnr_after": 27.52,
  "psnr_improvement": 1.67,
  "ssim_before": 0.9818,
  "ssim_after": 0.9876,
  "ssim_improvement": 0.0058,
  "mse_before": 0.0026,
  "mse_after": 0.0025,
  "sharpness_before": 0.1175,
  "sharpness_after": 0.0594
}
```

**Additional fields:**
- `*_before`: Metric on degraded input
- `*_after`: Metric on enhanced output
- `*_improvement`: Difference (positive = improvement)

---

## Quality Assessment

### Visual Inspection Checklist

After processing, check:

✅ **Good Results:**
- Edges are clearer and sharper
- Noise is reduced
- No artificial patterns or halos
- Details are preserved
- Natural appearance

❌ **Bad Results:**
- White or black images (clipping)
- Checkerboard/grid artifacts
- Halos around edges
- Over-smoothed (plastic-looking)
- Noise amplification

### Metric-Based Assessment

#### Scenario 1: Normal Microscopy (No Ground Truth)

**Good enhancement:**
```json
{
  "sharpness_after": 0.085,     // Increased from typical ~0.05
  "laplacian_after": 250.5      // Good focus measure
}
```

**Poor enhancement:**
```json
{
  "sharpness_after": 0.012,     // Too low - over-smoothed
  "laplacian_after": 15.3       // Very low - blurred
}
```

or

```json
{
  "sharpness_after": 0.450,     // Too high - noise amplified
  "laplacian_after": 2500.8     // Suspiciously high
}
```

#### Scenario 2: Synthetic Data (With Ground Truth)

**Good enhancement:**
```json
{
  "psnr_improvement": 1.67,     // Positive gain
  "ssim_improvement": 0.0058,   // Improved structure
  "psnr_after": 27.52,          // Good absolute quality
  "ssim_after": 0.9876          // High similarity
}
```

**Poor enhancement:**
```json
{
  "psnr_improvement": -0.47,    // ⚠️ NEGATIVE - quality degraded!
  "ssim_improvement": -0.0586,  // ⚠️ Structure destroyed
  "psnr_after": 14.81,          // Low quality
  "ssim_after": 0.5877          // Poor similarity
}
```

---

## Common Patterns

### Pattern 1: Over-Sharpening

**Symptoms:**
```json
{
  "sharpness_after": 0.350,      // Very high
  "laplacian_after": 1200.5,
  "ssim_improvement": -0.05      // Structure degraded
}
```

**Visual:** Halos, artificial edges, noise amplification

**Solution:** Reduce `amount` parameter in sharpening module

### Pattern 2: Over-Denoising

**Symptoms:**
```json
{
  "sharpness_after": 0.015,      // Very low
  "laplacian_after": 25.3,
  "psnr_improvement": -2.1       // Quality loss
}
```

**Visual:** Plastic/cartoon appearance, loss of texture

**Solution:** 
- Reduce denoising strength
- Use `BilateralFilter` instead of `NonLocalMeans`
- Lower `sigma_color` parameter

### Pattern 3: No Improvement

**Symptoms:**
```json
{
  "sharpness_after": 0.052,      // Unchanged
  "sharpness_before": 0.051,
  "psnr_improvement": 0.02       // Minimal change
}
```

**Visual:** Output looks identical to input

**Solution:**
- Increase `amount` in sharpening
- Use `aggressive.yaml` preset
- Check if input is already high-quality

### Pattern 4: Good Enhancement

**Symptoms:**
```json
{
  "psnr_improvement": 1.5,       // Solid gain
  "ssim_improvement": 0.012,     // Structure improved
  "sharpness_after": 0.095,      // Reasonable sharpness
  "laplacian_after": 285.3       // Good focus
}
```

**Visual:** Clearer edges, less noise, natural appearance

**Action:** Use these settings for remaining images

---

## Troubleshooting Results

### Issue 1: Images Are All White

**Cause:** Over-enhancement clipping to maximum value

**Check metrics:**
```json
{
  "sharpness_after": 0.0,        // Zero - all pixels same
  "laplacian_after": 0.0
}
```

**Solution:**
1. Use `gentle.yaml` preset
2. Reduce `amount` to 0.5-1.0
3. Check if input images are valid

### Issue 2: Images Are Blurry

**Cause:** Too much denoising

**Check metrics:**
```json
{
  "sharpness_after": 0.01,       // Very low
  "laplacian_after": 20.5
}
```

**Solution:**
1. Disable denoising: `enabled: false`
2. Use `GaussianDenoising` with low `sigma`
3. Add stronger sharpening after denoising

### Issue 3: Inconsistent Results

**Symptoms:** Some images good, others poor

**Check:** Individual JSON files for outliers

**Solution:**
1. Sort by metric: `sharpness_after`
2. Identify problematic images
3. May need different presets for different image types

### Issue 4: Processing Too Slow

**Check `overall_report.json`:**
```json
{
  "total_images": 100,
  "timestamp": "..." // Compare start/end times
}
```

**Solution:**
1. Use `BilateralFilter` (not `NonLocalMeans`)
2. Reduce `search_window_size`
3. Disable metrics: `enabled: false`
4. Use multi-threading (future feature)

---

## Comparing Presets

Use the comparison script:

```bash
python scripts/compare_presets.py
```

**Output:**
```
================================================================================
PRESET COMPARISON - Resolution Enhancement Microscopy
================================================================================

Preset           PSNR Δ (dB)       SSIM Δ     Final PSNR     Final SSIM
--------------------------------------------------------------------------------
aggressive             1.30       0.0320         21.78 dB        0.9166
default_v0             0.12      -0.0105         20.60 dB        0.8741
gentle                 0.14       0.0040         20.62 dB        0.8886

================================================================================
BEST PERFORMERS
================================================================================
Best PSNR Improvement: aggressive (+1.30 dB)
Best SSIM Improvement: aggressive (+0.0320)
Best Final PSNR:       aggressive (21.78 dB)
Best Final SSIM:       aggressive (0.9166)
================================================================================
```

**Interpretation:**
- **aggressive** gave best quantitative improvement (+1.30 dB)
- **default_v0** actually degraded structure (-0.0105 SSIM)
- **gentle** preserved structure better than default

**Conclusion:** Use aggressive preset for this dataset

---

## Programmatic Analysis

### Python Example: Analyzing Results

```python
import json
from pathlib import Path

# Load overall report
report_path = Path("results/default_v0/overall_report.json")
with open(report_path) as f:
    report = json.load(f)

# Print summary
print(f"Processed {report['total_images']} images")
print(f"Average sharpness: {report['avg_sharpness']:.4f}")
print(f"Average laplacian: {report['avg_laplacian']:.2f}")

# Find outliers
results = report['individual_results']
sorted_by_sharpness = sorted(results, key=lambda x: x['sharpness_after'])

print("\nLowest sharpness images:")
for r in sorted_by_sharpness[:5]:
    print(f"  {r['image']}: {r['sharpness_after']:.4f}")

print("\nHighest sharpness images:")
for r in sorted_by_sharpness[-5:]:
    print(f"  {r['image']}: {r['sharpness_after']:.4f}")
```

---

## Summary

### Quick Reference Table

| Metric | Good Value | Bad Value | Indicates |
|--------|-----------|-----------|-----------|
| **Sharpness** | 0.05-0.30 | < 0.02 or > 0.40 | Edge clarity |
| **Laplacian** | 100-500 | < 30 or > 1000 | Focus quality |
| **PSNR** | > 25 dB | < 20 dB | Pixel accuracy |
| **SSIM** | > 0.90 | < 0.70 | Perceptual quality |
| **PSNR Δ** | > +1.0 dB | < 0 dB (negative) | Improvement |
| **SSIM Δ** | > +0.01 | < 0 (negative) | Structure preservation |

### Decision Tree

```
Is enhancement visible?
├─ No → Increase sharpening amount
└─ Yes → Are there artifacts?
    ├─ Yes (halos, noise) → Reduce amount or use gentle preset
    └─ No → Check metrics
        ├─ Sharpness improved? 
        │   ├─ Yes → ✓ Good result
        │   └─ No → Try different preset
        └─ PSNR/SSIM improved?
            ├─ Yes → ✓ Good result
            └─ No → Over-processing, use less aggressive preset
```

---

## Next Steps

- **Optimize Settings**: Return to [Configuring Presets](02_Configuring_Presets.md)
- **Learn More**: See [Enhancement Algorithms](04_Enhancement_Algorithms.md)
- **Process Full Dataset**: Once satisfied with test results

---

**Version:** v0  
**Last Updated:** November 2, 2025  
**Author:** Mehul Yadav
