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

Reson provides **3 deconvolution algorithms**, each with different strengths:

### 1. Richardson-Lucy (RL)

**Type:** Iterative, maximum likelihood

**Best for:**
- Fluorescence microscopy
- Poisson noise (photon counting)
- General-purpose deconvolution

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
- `regularization`: 0.0-0.01 (prevents negative values)

**Characteristics:**
- ✅ Physically motivated (Poisson statistics)
- ✅ Preserves non-negativity
- ✅ Handles Poisson noise well
- ⚠️ Can over-sharpen if too many iterations
- ⚠️ Slower (iterative)

---

### 2. Wiener Deconvolution

**Type:** Frequency domain, closed-form solution

**Best for:**
- Fast processing
- Gaussian noise
- When you know noise level

**Algorithm:**
```
U(f) = [H*(f) / (|H(f)|² + K)] · Y(f)
```

Where:
- `U(f)` = deconvolved image (frequency domain)
- `H(f)` = PSF (frequency domain)
- `Y(f)` = observed image (frequency domain)
- `K` = noise power parameter

**Parameters:**
- `noise_power`: 0.001-0.1 (regularization parameter)

**Characteristics:**
- ✅ Very fast (single FFT operation)
- ✅ Good for Gaussian noise
- ✅ Closed-form solution
- ❌ Can produce negative values
- ❌ Less effective for Poisson noise
- ❌ May introduce ringing

---

### 3. Total Variation (TV) Deconvolution

**Type:** Regularized optimization

**Best for:**
- Edge preservation
- Piecewise smooth images
- Strong noise

**Algorithm:**
```
minimize: ||PSF ⊗ u - y||² + λ · TV(u)
```

Where:
- `TV(u)` = Total Variation (sum of gradients)
- `λ` = regularization strength

**Parameters:**
- `iterations`: 50-200
- `lambda_tv`: 0.001-0.1 (edge preservation vs smoothness)

**Characteristics:**
- ✅ Preserves edges
- ✅ Reduces noise
- ✅ Good for piecewise smooth images
- ⚠️ Can create "staircase" artifacts
- ⚠️ Slower (iterative optimization)
- ⚠️ May over-smooth fine details

---

## Algorithm Comparison

### Quick Comparison Table

| Feature | Richardson-Lucy | Wiener | Total Variation |
|---------|-----------------|--------|-----------------|
| **Speed** | ⚡⚡ (slow) | ⚡⚡⚡⚡ (fast) | ⚡ (slowest) |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Noise type** | Poisson | Gaussian | Both |
| **Iterations** | 10-50 | 1 (direct) | 50-200 |
| **Edge preservation** | Good | Poor | Excellent |
| **Artifacts** | Over-sharpening | Ringing | Staircase |
| **Use case** | Fluorescence | Fast preview | Noisy images |

### Performance Benchmarks

Tested on 512×512 synthetic images:

| Algorithm | PSF Size | Iterations | Time | Quality |
|-----------|----------|------------|------|---------|
| **Richardson-Lucy** | 31×31 | 20 | ~1.5s | Excellent |
| **Richardson-Lucy** | 41×41 | 30 | ~12s | Excellent |
| **Wiener** | 31×31 | 1 | ~0.1s | Good |
| **Total Variation** | 31×31 | 100 | ~5s | Very Good |

*Tested on: Windows 11, Python 3.13, no GPU acceleration*

### Decision Tree

```
What type of noise do you have?
├─ Poisson (photon counting, fluorescence)
│  └─ Use Richardson-Lucy
├─ Gaussian (camera noise, uniform)
│  ├─ Need speed? → Use Wiener
│  └─ Need quality? → Use Richardson-Lucy or TV
└─ Heavy noise + edges important
   └─ Use Total Variation
```

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
| **Fluorescence (widefield)** | Richardson-Lucy + Gibson-Lanni | `deconv_fluorescence.yaml` |
| **Fluorescence (general)** | Richardson-Lucy + Gaussian | `deconv_rl.yaml` |
| **Brightfield** | Wiener + Airy | `deconv_wiener.yaml` |
| **Noisy images** | TV + appropriate PSF | `deconv_tv.yaml` |

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
2. **Not sharp enough** → Increase iterations, try different PSF
3. **Ringing artifacts** → Reduce iterations, check PSF accuracy
4. **Too slow** → Reduce PSF size, reduce iterations, try Wiener

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

Available deconvolution modules:

```yaml
type: "RichardsonLucy"      # Iterative, Poisson noise
type: "WienerDeconvolution" # Fast, frequency domain
type: "TVDeconvolution"     # Edge-preserving, regularized
```

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

#### Method 4: Custom PSF (Measured)

```yaml
params:
  psf_method: "custom"
  psf_params:
    psf_file: "data/measured_psf.tif"  # or .npy
```

#### Method 5: Blind PSF (Estimated)

```yaml
params:
  psf_method: "blind"
  psf_params:
    method: "autocorrelation"  # or "edge_based", "cepstrum"
    psf_size: 31
```

### Richardson-Lucy Parameters

```yaml
type: "RichardsonLucy"
params:
  psf_method: "gaussian"
  psf_params: { ... }
  iterations: 20              # Key parameter!
  regularization: 0.001       # Prevent negative values
```

**Tuning:**
- `iterations`:
  - **10-15**: Mild deconvolution, safe
  - **20-30**: Moderate, good balance
  - **40-50**: Aggressive, may over-sharpen
  - **>50**: Likely to produce artifacts

- `regularization`:
  - **0.0**: No regularization (may produce negative values)
  - **0.001**: Light (recommended)
  - **0.01**: Heavy (reduces sharpness slightly)

### Wiener Parameters

```yaml
type: "WienerDeconvolution"
params:
  psf_method: "airy"
  psf_params: { ... }
  noise_power: 0.01           # Key parameter!
```

**Tuning:**
- `noise_power`:
  - **0.001**: Low noise, aggressive deconvolution
  - **0.01**: Moderate (recommended starting point)
  - **0.1**: High noise, conservative

### Total Variation Parameters

```yaml
type: "TVDeconvolution"
params:
  psf_method: "gaussian"
  psf_params: { ... }
  iterations: 100             # More than RL
  lambda_tv: 0.01             # Regularization strength
```

**Tuning:**
- `iterations`:
  - **50**: Fast, may not converge
  - **100**: Good balance (recommended)
  - **200**: Careful optimization

- `lambda_tv`:
  - **0.001**: Weak regularization, sharper
  - **0.01**: Moderate (recommended)
  - **0.1**: Strong, may over-smooth

---

## Parameter Tuning

### Strategy 1: Start Conservative

Begin with safe parameters, then increase:

**Richardson-Lucy:**
```yaml
iterations: 15              # Start low
regularization: 0.001       # Light regularization
```

**Wiener:**
```yaml
noise_power: 0.01          # Moderate
```

**Total Variation:**
```yaml
iterations: 100
lambda_tv: 0.01
```

### Strategy 2: Binary Search

If too blurry → increase sharpness
If too sharp/artifacts → decrease sharpness

**Example (Richardson-Lucy iterations):**
1. Try 20 → too blurry
2. Try 40 → too sharp, ringing
3. Try 30 → good!

### Strategy 3: Test on Synthetic Data

Use provided test data to find good parameters:

```bash
# Test different iteration counts
for i in 10 20 30 40 50; do
    # Modify config to use $i iterations
    python main.py \
        -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \
        -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \
        -c configs/deconv_rl.yaml
    # Compare metrics
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
- ✅ Reduce iterations
- ✅ Use Wiener (fastest algorithm)
- ✅ Crop region of interest before processing

**Performance tips:**
```yaml
# Fast config
psf_params:
  size: 21              # Smaller PSF
iterations: 15          # Fewer iterations
```

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

**Scenario:** Brightfield, white light, 40× objective, need fast processing

**Config:**
```yaml
version: v1

enhancement:
  modules:
    - name: "Wiener Deconvolution"
      type: "WienerDeconvolution"
      enabled: true
      params:
        psf_method: "airy"
        psf_params:
          wavelength: 550           # Green light (middle of spectrum)
          numerical_aperture: 0.75  # 40× dry objective
          pixel_size: 0.163         # 6.5µm camera, 40× objective
          size: 31
        noise_power: 0.01

# ... rest of config
```

**Run:**
```bash
python main.py -i brightfield.tif -c configs/brightfield_fast.yaml
```

**Expected:**
- Fast processing (~0.1-0.2s)
- Moderate sharpening
- Good for previews or high-throughput

---

### Example 3: Noisy Fluorescence (Edge-Preserving)

**Scenario:** Low-light fluorescence, noisy, want to preserve edges

**Config:**
```yaml
version: v1

enhancement:
  modules:
    - name: "TV Deconvolution"
      type: "TVDeconvolution"
      enabled: true
      params:
        psf_method: "gaussian"
        psf_params:
          wavelength: 610           # RFP emission
          numerical_aperture: 1.2   # Water immersion
          pixel_size: 0.108         # 6.5µm camera, 60× objective
          size: 31
        iterations: 100
        lambda_tv: 0.01

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

**Use provided test data to validate:**

```bash
# Richardson-Lucy on checkerboard
python main.py \
    -i data/synthetic_psf/checkerboard_gaussian_mild/blurred.tif \
    -g data/synthetic_psf/checkerboard_gaussian_mild/ground_truth.tif \
    -c configs/deconv_rl.yaml \
    --verbose

# Wiener on star pattern
python main.py \
    -i data/synthetic_psf/star_airy/blurred.tif \
    -g data/synthetic_psf/star_airy/ground_truth.tif \
    -c configs/deconv_wiener.yaml \
    --verbose

# Fluorescence beads
python main.py \
    -i data/synthetic_psf/fluorescent_beads_fluorescence_poisson5/blurred.tif \
    -g data/synthetic_psf/fluorescent_beads_fluorescence_poisson5/ground_truth.tif \
    -c configs/deconv_fluorescence.yaml \
    --verbose
```

**Compare metrics to evaluate:**
- PSNR, SSIM (with ground truth)
- Sharpness improvement
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
- Richardson, W.H. (1972). "Bayesian-Based Iterative Method of Image Restoration"
- Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions"

### Wiener Filter
- Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing of Stationary Time Series"

### Total Variation
- Rudin, L.I., Osher, S., Fatemi, E. (1992). "Nonlinear total variation based noise removal algorithms"

---

**Previous:** [PSF Generation Guide](05_PSF_Generation.md)
**Next:** [Enhancement Algorithms](04_Enhancement_Algorithms.md)
