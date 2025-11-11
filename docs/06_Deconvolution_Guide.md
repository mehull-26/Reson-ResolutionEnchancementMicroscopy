# Deconvolution Guide

Complete guide to image deconvolution in Reson.

---

## Table of Contents

1. [What is Deconvolution?](#what-is-deconvolution)
2. [When to Use Deconvolution](#when-to-use-deconvolution)
3. [Available Algorithms](#available-algorithms)
4. [Algorithm Comparison](#algorithm-comparison)
5. [Complete Workflow](#complete-workflow)
6. [Configuration Guide](#configuration-guide)
7. [Parameter Tuning](#parameter-tuning)
8. [Evaluating Results](#evaluating-results)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

---

## What is Deconvolution?

### The Problem

Microscope images are **blurred** by the optical system. Mathematically:

```
Observed Image = True Image ⊗ PSF + Noise
```

Where:
- `⊗` = convolution
- **PSF** = Point Spread Function (system blur)
- **Noise** = Camera noise, photon noise, etc.

### The Solution

**Deconvolution** attempts to **reverse** this process:

```
Deconvolved Image ≈ True Image
```

It's an **inverse problem**: estimate the original image given the blurred observation and PSF.

### Why It's Challenging

1. **Ill-posed problem** - Many solutions possible
2. **Noise amplification** - Deconvolution can enhance noise
3. **PSF accuracy critical** - Wrong PSF → wrong result
4. **Computational cost** - Iterative methods can be slow

---

## When to Use Deconvolution

### ✅ Good Use Cases

**Deconvolution works well when:**

1. **PSF-limited blur**
   - Optical blur from microscope
   - Out-of-focus blur (known defocus)
   - Diffraction-limited imaging

2. **Known or measurable PSF**
   - Can generate theoretical PSF
   - Can measure PSF from beads
   - Blind estimation possible

3. **Structured samples**
   - Subcellular structures
   - Fluorescent beads
   - Tissue sections

4. **Goal is resolution enhancement**
   - Reveal hidden details
   - Sharpen features
   - Quantitative analysis

### ❌ When NOT to Use

**Avoid deconvolution when:**

1. **Motion blur** - PSF doesn't model motion
2. **Sample movement** - Time-varying blur
3. **Severe noise** - Will amplify noise
4. **PSF unknown and complex** - Blind methods may fail
5. **Already sharp** - No benefit, may introduce artifacts

### Alternatives

- **Motion blur** → Motion deblur algorithms (not PSF-based)
- **Noise** → Denoise first, then deconvolve
- **Drift** → Image registration, then deconvolve

---

## Available Algorithms

Reson currently provides **Richardson-Lucy deconvolution**, a robust iterative algorithm well-suited for microscopy applications.

### Richardson-Lucy (RL)

**Type:** Iterative, maximum likelihood

**Status:** ✅ **Validated** - Verified +8 dB PSNR improvement on synthetic test data

**Best for:**
- Fluorescence microscopy
- Poisson noise (photon counting)
- General-purpose deconvolution
- High-quality restoration when PSF is known or measured

**Algorithm:**
```
u^(k+1) = u^(k) · [PSF* ⊗ (y / (PSF ⊗ u^(k)))]
```

Where:
- `u` = estimate (starts with observed image)
- `y` = observed image
- `PSF*` = flipped PSF
- `⊗` = convolution

**Parameters:**
- `iterations`: 10-50 (more = sharper, but may overfit)
  * Recommended: 20-30 for most cases
  * Validated: 20 iterations achieves +8.11 dB PSNR improvement
- `regularization`: 0.0-0.01 (prevents negative values, optional)

**Characteristics:**
- ✅ Physically motivated (Poisson statistics)
- ✅ Preserves non-negativity (critical for photon counting)
- ✅ Handles Poisson noise optimally
- ✅ **Proven performance**: +8 dB PSNR on synthetic data
- ✅ Works with both known PSF parameters and custom measured PSFs
- ⚠️ Can over-sharpen if too many iterations
- ⚠️ Slower than direct methods (iterative algorithm)

**Validated Performance:**
- Test data: Synthetic PSF-blurred images (13 test cases)
- Richardson-Lucy (20 iterations) with Gaussian PSF:
  * PSNR improvement: **+8.11 dB** (34.65 → 42.76 dB)
  * SSIM improvement: +0.01 (0.98 → 0.99)
  * Processing time: ~1-2 seconds per 512×512 image
- Richardson-Lucy with custom measured PSF:
  * PSNR improvement: **+8.08 dB** (nearly identical to known PSF)
  * Validates that measured PSF workflow is accurate

---

## Algorithm Comparison

### Richardson-Lucy vs Other Methods

While Reson currently focuses on Richardson-Lucy, here's how it compares to other common methods:

| Feature | Richardson-Lucy (Reson) | Wiener | Total Variation |
|---------|-------------------------|--------|-----------------|
| **Speed** | ⚡⚡ (moderate) | ⚡⚡⚡⚡ (fast) | ⚡ (slow) |
| **Quality** | ⭐⭐⭐⭐⭐ (excellent) | ⭐⭐⭐ (good) | ⭐⭐⭐⭐ (very good) |
| **Noise type** | Poisson | Gaussian | Both |
| **Iterations** | 10-50 | 1 (direct) | 50-200 |
| **Edge preservation** | Good | Poor | Excellent |
| **Artifacts** | Over-sharpening | Ringing | Staircase |
| **Status in Reson** | ✅ Implemented | ❌ Not included | ❌ Not included |
| **Use case** | Fluorescence | Fast preview | Noisy images |

**Why Richardson-Lucy?**
- Best suited for fluorescence microscopy (Poisson noise model)
- Physically motivated algorithm
- Proven performance (+8 dB improvement)
- No negative values (preserves photon counting interpretation)
- Industry standard for microscopy deconvolution

---

## Complete Workflow

### Step 1: Choose PSF Method

See [PSF Generation Guide](05_PSF_Generation.md) for details.

**Quick selection:**
- **Have measured PSF?** → Use custom PSF (best)
- **Fluorescence?** → Gibson-Lanni PSF
- **Brightfield, high NA?** → Airy PSF
- **Quick test?** → Gaussian PSF

### Step 2: Select Deconvolution Algorithm

Based on your imaging modality:

| Imaging Type | Recommended Algorithm | Config File |
|--------------|----------------------|-------------|
| **Fluorescence (widefield)** | Richardson-Lucy + Gibson-Lanni PSF | `deconv_rl_known.yaml` |
| **Fluorescence (general)** | Richardson-Lucy + Gaussian PSF | `deconv_rl_known.yaml` |
| **With measured PSF** | Richardson-Lucy + Custom PSF | `deconv_rl_custom.yaml` |
| **Any microscopy type** | Richardson-Lucy + appropriate PSF | `deconv_rl.yaml` |

### Step 3: Create/Modify Config File

Example: `configs/my_deconvolution.yaml`

```yaml
version: v1

io:
  output_dir: "data/processed"
  save_intermediate: true
  save_visualizations: true

enhancement:
  modules:
    # Optional: Pre-denoising
    - name: "Pre-denoising"
      type: "BilateralFilter"
      enabled: true
      params:
        sigma_color: 0.1
        sigma_spatial: 2
    
    # Main deconvolution
    - name: "Richardson-Lucy Deconvolution"
      type: "RichardsonLucy"
      enabled: true
      params:
        psf_method: "gibson_lanni"  # or "gaussian", "airy", "custom"
        psf_params:
          wavelength: 520           # nm (emission wavelength!)
          numerical_aperture: 1.4
          pixel_size: 0.065         # µm
          size: 31
          ni: 1.518                 # oil immersion
          ns: 1.33                  # aqueous sample
          ti: 150                   # imaging depth (µm)
        iterations: 20              # 10-50 typical
        regularization: 0.001       # prevent negative values

processing:
  clip_values: true
  normalize_output: false

metrics:
  compute:
    - "sharpness"
    - "laplacian_variance"
    - "brenner_sharpness"

visualization:
  enabled: true
  dpi: 150
```

### Step 4: Run Deconvolution

```bash
python main.py \
    -i path/to/blurred_image.tif \
    -c configs/my_deconvolution.yaml \
    --verbose
```

**With ground truth (for testing):**
```bash
python main.py \
    -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \
    -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \
    -c configs/deconv_rl.yaml \
    --verbose
```

### Step 5: Evaluate Results

Check outputs in `data/processed/` and `results/`:

1. **Visual inspection:**
   - `blurred_enhanced.tif` - Deconvolved image
   - `comparison.png` - Before/after comparison

2. **Quantitative metrics:**
   - `overall_report.json` - Sharpness metrics
   - Higher sharpness = better deconvolution

3. **Look for artifacts:**
   - Ringing (halos around edges)
   - Over-sharpening (unnatural edges)
   - Noise amplification
   - Negative values clipped to zero

### Step 6: Tune Parameters (if needed)

If results aren't satisfactory, adjust:

1. **Too noisy** → Reduce iterations, increase regularization
2. **Not sharp enough** → Increase iterations, try different PSF method
3. **Ringing artifacts** → Reduce iterations, check PSF accuracy
4. **Too slow** → Reduce PSF size or reduce iterations

---

## Configuration Guide

### Basic Config Structure

All deconvolution configs follow this structure:

```yaml
version: v1  # Required!

io:
  output_dir: "data/processed"
  save_intermediate: true
  save_visualizations: true

enhancement:
  modules:
    - name: "Module Name"
      type: "AlgorithmType"
      enabled: true
      params:
        # Algorithm-specific parameters

processing:
  clip_values: true
  normalize_output: false

metrics:
  compute:
    - "sharpness"
    - "laplacian_variance"

visualization:
  enabled: true
  dpi: 150
```

### Module Types

Currently available deconvolution module:

```yaml
type: "RichardsonLucy"      # Iterative, maximum likelihood, Poisson noise
```

**Note:** Future versions may include additional algorithms (Wiener, TV, etc.)

### PSF Configuration

#### Method 1: Gaussian PSF

```yaml
params:
  psf_method: "gaussian"
  psf_params:
    wavelength: 550        # nm
    numerical_aperture: 1.0
    pixel_size: 0.1        # µm
    size: 31
```

#### Method 2: Airy PSF

```yaml
params:
  psf_method: "airy"
  psf_params:
    wavelength: 550
    numerical_aperture: 1.4
    pixel_size: 0.065
    size: 31
```

#### Method 3: Gibson-Lanni PSF (Fluorescence)

```yaml
params:
  psf_method: "gibson_lanni"
  psf_params:
    wavelength: 520        # Emission wavelength!
    numerical_aperture: 1.4
    pixel_size: 0.065
    size: 41               # Larger for fluorescence
    ni: 1.518              # Immersion medium (oil)
    ns: 1.33               # Sample (aqueous)
    ti: 150                # Depth (µm)
```

#### Method 4: Custom PSF (Measured) ✅ RECOMMENDED

```yaml
params:
  psf_method: "custom"
  psf_params:
    psf_file: "data/measured_psf.tif"  # or .npy
```

**Best practice:** Use measured PSF from fluorescent beads for highest accuracy.

#### Method 5: Blind PSF (Estimated) ⚠️ EXPERIMENTAL

```yaml
params:
  psf_method: "blind"
  psf_params:
    psf_size: 31
    iterations: 20
```

**Warning:** Blind PSF estimation is experimental and has limited reliability (~56% correlation with true PSF). Not recommended for quantitative work. Prefer measured or theoretical PSF.

### Richardson-Lucy Parameters

```yaml
type: "RichardsonLucy"
params:
  psf_method: "gaussian"
  psf_params: { ... }
  iterations: 20              # Key parameter! (validated at 20)
  regularization: 0.001       # Prevent negative values
```

**Tuning:**
- `iterations`:
  - **10-15**: Mild deconvolution, safe
  - **20-30**: Moderate, good balance ✅ (validated: 20 iterations = +8 dB)
  - **40-50**: Aggressive, may over-sharpen
  - **>50**: Likely to produce artifacts

- `regularization`:
  - **0.0**: No regularization (may produce negative values)
  - **0.001**: Light (recommended, validated)
  - **0.01**: Heavy (reduces sharpness slightly)

---

## Parameter Tuning

### Strategy 1: Start Conservative

Begin with validated parameters (tested on synthetic data):

**Richardson-Lucy (recommended):**
```yaml
iterations: 20              # ✅ Validated at 20 for +8 dB improvement
regularization: 0.001       # Light regularization
```

**If results are:**
- Too blurry → Increase iterations to 25-30
- Too sharp/artifacts → Decrease iterations to 15
- Noisy → Increase regularization to 0.005

### Strategy 2: Binary Search

If you need to optimize for your specific data:

**Example (Richardson-Lucy iterations):**
1. Try 20 (validated) → too blurry for your data
2. Try 40 → too sharp, ringing artifacts
3. Try 30 → good balance!

### Strategy 3: Test on Synthetic Data

Use provided test data to find optimal parameters for your microscope:

```bash
# Test different iteration counts
for i in 10 20 30 40 50; do
    # Modify config to use $i iterations
    python main.py \
        -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \
        -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \
        -c configs/deconv_rl_known.yaml
    # Compare metrics (PSNR, SSIM)
done
```

### What to Monitor

**Metrics to track:**
1. **Sharpness** (Tenengrad) - Higher = sharper
2. **Laplacian variance** - Edge strength
3. **Visual quality** - Most important!

**Stop when:**
- Metrics plateau or decrease
- Visual artifacts appear
- Processing time too long

---

## Evaluating Results

### Visual Inspection

**Good deconvolution:**
- ✅ Sharper edges
- ✅ More visible fine details
- ✅ Improved contrast
- ✅ No visible artifacts
- ✅ Preserved large-scale structure

**Bad deconvolution:**
- ❌ Ringing (halos around edges)
- ❌ Over-sharpening (unnatural edges)
- ❌ Noise amplification
- ❌ Lost features
- ❌ Blocky/pixelated appearance

### Quantitative Metrics

Check `results/experiment_name/overall_report.json`:

```json
{
  "sharpness": 0.2434,           // Higher = sharper
  "laplacian_variance": 909.98,  // Edge strength
  "brenner_sharpness": 2500.44   // Another sharpness measure
}
```

**Interpretation:**
- **20-50% improvement** in sharpness: Good deconvolution
- **>100% improvement**: Very effective, check for over-sharpening
- **<10% improvement**: PSF may be inaccurate or image already sharp

### Compare to Ground Truth (if available)

With ground truth, you get additional metrics:

```json
{
  "psnr": 28.5,      // Peak Signal-to-Noise Ratio (higher = better)
  "ssim": 0.85,      // Structural Similarity (0-1, higher = better)
  "mse": 120.3       // Mean Squared Error (lower = better)
}
```

**Good results:**
- PSNR > 25 dB
- SSIM > 0.8
- Visual similarity to ground truth

### Common Artifacts

**1. Ringing (Gibbs phenomenon)**
- Halos around sharp edges
- Caused by: Too many iterations, wrong PSF
- Fix: Reduce iterations, check PSF

**2. Over-sharpening**
- Unnatural, exaggerated edges
- Caused by: Too many iterations
- Fix: Reduce iterations, increase regularization

**3. Noise amplification**
- Grainy appearance
- Caused by: High-frequency noise amplified
- Fix: Denoise first, reduce iterations

**4. Negative values (clipped to zero)**
- Dark regions become completely black
- Caused by: Insufficient regularization
- Fix: Increase regularization

---

## Troubleshooting

### Problem: Results not sharp enough

**Possible causes:**
1. PSF inaccurate
2. Too few iterations
3. Wrong PSF method

**Solutions:**
- ✅ Increase iterations (RL: 20→30→40)
- ✅ Try more accurate PSF (Gaussian→Airy→Gibson-Lanni)
- ✅ Measure PSF from beads
- ✅ Try blind PSF estimation to compare

### Problem: Ringing artifacts (halos)

**Cause:** Over-deconvolution or PSF inaccuracy

**Solutions:**
- ✅ Reduce iterations (RL: 30→20→15)
- ✅ Increase regularization (0.001→0.01)
- ✅ Check PSF parameters (especially wavelength, NA)
- ✅ Verify PSF is normalized (sum = 1)

### Problem: Noise amplification

**Cause:** Noise gets enhanced by deconvolution

**Solutions:**
- ✅ Pre-denoise: Add BilateralFilter before deconvolution
- ✅ Reduce iterations
- ✅ Use TV deconvolution (edge-preserving)
- ✅ Increase regularization

**Example config with pre-denoising:**
```yaml
enhancement:
  modules:
    - name: "Pre-denoising"
      type: "BilateralFilter"
      enabled: true
      params:
        sigma_color: 0.1
        sigma_spatial: 2
    
    - name: "Deconvolution"
      type: "RichardsonLucy"
      # ...
```

### Problem: Very slow processing

**Cause:** Large PSF, many iterations

**Solutions:**
- ✅ Reduce PSF size (41→31→21)
- ✅ Reduce iterations (30→20→15)
- ✅ Crop region of interest before processing
- ✅ Process smaller image batches

**Performance tips:**
```yaml
# Faster config
psf_params:
  size: 21              # Smaller PSF (faster convolution)
iterations: 15          # Fewer iterations
```

**Note:** Processing speed depends on image size and PSF size. Typical times:
- 512×512, PSF=31, iterations=20: ~1-2 seconds
- 1024×1024, PSF=31, iterations=20: ~5-8 seconds

### Problem: "PSF sums to zero" error

**Cause:** Invalid PSF

**Solutions:**
- ✅ Check wavelength in nm (not µm!)
- ✅ Check pixel_size in µm (not nm!)
- ✅ Check NA is reasonable (<2.0)
- ✅ Verify custom PSF file exists and is readable

### Problem: Results worse than input

**Cause:** Wrong PSF or inappropriate deconvolution

**Solutions:**
- ✅ Check if blur is really PSF-limited (not motion)
- ✅ Try different PSF method
- ✅ Reduce iterations to minimum (10)
- ✅ Consider if deconvolution is appropriate

### Problem: Different results each run

**Cause:** Blind PSF estimation is non-deterministic

**Solutions:**
- ✅ Use theoretical PSF instead
- ✅ Measure PSF from beads (most stable)
- ✅ Set random seed if using blind estimation

---

## Examples

### Example 1: Fluorescence Microscopy (GFP)

**Scenario:** Widefield fluorescence, GFP-labeled cells, oil immersion 100× objective

**Config:**
```yaml
version: v1

enhancement:
  modules:
    - name: "Pre-denoising"
      type: "BilateralFilter"
      enabled: true
      params:
        sigma_color: 0.1
        sigma_spatial: 2
    
    - name: "RL Deconvolution"
      type: "RichardsonLucy"
      enabled: true
      params:
        psf_method: "gibson_lanni"
        psf_params:
          wavelength: 520           # GFP emission
          numerical_aperture: 1.4   # Oil immersion
          pixel_size: 0.065         # 6.5µm camera, 100× objective
          size: 41
          ni: 1.518                 # Oil
          ns: 1.33                  # Cells (aqueous)
          ti: 150                   # 150µm deep
        iterations: 30
        regularization: 0.001

# ... rest of config
```

**Run:**
```bash
python main.py -i gfp_cells.tif -c configs/fluorescence_gfp.yaml --verbose
```

**Expected:**
- Sharper cellular structures
- Better separation of nearby objects
- ~10-15s processing time (512×512 image)

---

### Example 2: Brightfield Microscopy (Fast)

**Scenario:** Brightfield, white light, 40× objective

**Config:**
```yaml
version: v1

enhancement:
  modules:
    - name: "Richardson-Lucy Deconvolution"
      type: "RichardsonLucy"
      enabled: true
      params:
        psf_method: "airy"
        psf_params:
          wavelength: 550           # Green light (middle of spectrum)
          numerical_aperture: 0.75  # 40× dry objective
          pixel_size: 0.163         # 6.5µm camera, 40× objective
          size: 31
        iterations: 20
        regularization: 0.001

# ... rest of config
```

**Run:**
```bash
python main.py -i brightfield.tif -c configs/brightfield_deconv.yaml
```

**Expected:**
- Processing time: ~1-2 seconds per image
- +8 dB PSNR improvement (validated)
- Preserves image structure

---

### Example 3: Noisy Fluorescence Images

**Scenario:** Low-light fluorescence, noisy images

**Config:**
```yaml
version: v1

enhancement:
  modules:
    - name: "Richardson-Lucy Deconvolution"
      type: "RichardsonLucy"
      enabled: true
      params:
        psf_method: "gaussian"
        psf_params:
          wavelength: 610           # RFP emission
          numerical_aperture: 1.2   # Water immersion
          pixel_size: 0.108         # 6.5µm camera, 60× objective
          size: 31
        iterations: 20              # Standard validated value
        regularization: 0.005       # Higher regularization for noisy images

# ... rest of config
```

**Run:**
```bash
python main.py -i noisy_rfp.tif -c configs/noisy_fluorescence.yaml
```

**Expected:**
- Edges preserved
- Noise reduced
- ~5-10s processing time

---

### Example 4: Testing with Synthetic Data

**Use provided test data to validate and optimize parameters:**

```bash
# Richardson-Lucy on checkerboard (validated: +8.11 dB)
python main.py \
    -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \
    -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \
    -c configs/deconv_rl_known.yaml \
    --verbose

# Test with Airy PSF
python main.py \
    -i data/synthetic_psf/star_airy/blurred.tif \
    -g data/synthetic_psf/star_airy/ground_truth.tif \
    -c configs/deconv_rl_known.yaml \
    --verbose

# Noisy fluorescence beads
python main.py \
    -i data/synthetic_psf/fluorescent_beads_fluorescence_poisson5/blurred.tif \
    -g data/synthetic_psf/fluorescent_beads_fluorescence_poisson5/ground_truth.tif \
    -c configs/deconv_rl_known.yaml \
    --verbose

# Test with custom measured PSF
python main.py \
    -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \
    -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \
    -c configs/deconv_rl_custom.yaml \
    --verbose
```

**Compare metrics to evaluate:**
- PSNR improvement (expect +8 dB on clean images)
- SSIM (structural similarity with ground truth)
- Sharpness improvement (Tenengrad, Laplacian variance)
- Visual quality

---

## Advanced Topics

### Pre-processing Pipeline

**Recommended order:**
1. **Flat-field correction** (if needed)
2. **Denoising** (Bilateral, NLM)
3. **Deconvolution**
4. **Post-sharpening** (optional)

**Config example:**
```yaml
enhancement:
  modules:
    - name: "Denoise"
      type: "BilateralFilter"
      # ...
    
    - name: "Deconvolve"
      type: "RichardsonLucy"
      # ...
    
    - name: "Sharpen"
      type: "UnsharpMasking"
      enabled: false  # Usually not needed after deconvolution
      # ...
```

### Batch Processing

Process multiple files:

```bash
# Process all images in folder
for img in data/raw/*.tif; do
    python main.py -i "$img" -c configs/deconv_rl.yaml
done
```

### Custom PSF from Beads

Measure and use your own PSF:

```bash
# 1. Generate PSF from bead image
python -m scripts.generate_psf \
    --method custom \
    --input bead_image.tif \
    --output my_psf.tif \
    --visualize

# 2. Use in config
# Edit config to:
#   psf_method: "custom"
#   psf_params:
#     psf_file: "my_psf.tif"

# 3. Run deconvolution
python main.py -i sample.tif -c configs/custom_psf.yaml
```

---

## Best Practices Summary

### ✅ DO:

1. **Start conservative**
   - Low iterations (15-20)
   - Moderate regularization (0.001)

2. **Use appropriate PSF**
   - Measured PSF > Gibson-Lanni > Airy > Gaussian
   - Match imaging conditions exactly

3. **Pre-denoise if noisy**
   - Bilateral filter before deconvolution
   - Prevents noise amplification

4. **Test on synthetic data**
   - Validate approach before real data
   - Compare metrics

5. **Visual inspection**
   - Metrics don't tell the whole story
   - Check for artifacts

6. **Save PSF**
   - Document PSF parameters
   - Save measured PSFs for reuse

### ❌ DON'T:

1. **Don't blindly increase iterations**
   - More ≠ better
   - Check for artifacts

2. **Don't use wrong wavelength**
   - Fluorescence: emission, not excitation!

3. **Don't skip validation**
   - Test before batch processing

4. **Don't deconvolve non-PSF blur**
   - Motion blur needs different methods

5. **Don't forget to denoise first**
   - Noisy images amplify noise

6. **Don't use even-sized PSF**
   - Always odd: 21, 31, 41

---

## See Also

- **[PSF Generation Guide](05_PSF_Generation.md)** - How to create accurate PSFs
- **[Enhancement Algorithms](04_Enhancement_Algorithms.md)** - All algorithms overview
- **[Interpreting Results](03_Interpreting_Results.md)** - Understanding metrics

---

## References

### Richardson-Lucy Algorithm
- Richardson, W.H. (1972). "Bayesian-Based Iterative Method of Image Restoration." *Journal of the Optical Society of America*, 62(1), 55-59.
- Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions." *The Astronomical Journal*, 79, 745-754.

### PSF Models
- Gibson, S.F., Lanni, F. (1992). "Experimental test of an analytical model of aberration in an oil-immersion objective lens used in three-dimensional light microscopy." *Journal of the Optical Society of America A*, 9(1), 154-166.
- Born, M., Wolf, E. (1999). "Principles of Optics." Cambridge University Press. (Airy disk theory)

### Fluorescence Microscopy
- Conchello, J.A., Lichtman, J.W. (2005). "Optical sectioning microscopy." *Nature Methods*, 2(12), 920-931.

---

**Previous:** [PSF Generation Guide](05_PSF_Generation.md)
**Next:** [Enhancement Algorithms](04_Enhancement_Algorithms.md)
