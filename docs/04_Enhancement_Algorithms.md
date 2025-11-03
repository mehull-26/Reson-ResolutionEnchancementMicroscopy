# Enhancement Algorithms Reference

## Reson v0 - Technical Documentation

Complete reference for all enhancement algorithms implemented in Reson v0.

---

## Table of Contents
1. [Overview](#overview)
2. [Denoising Algorithms](#denoising-algorithms)
3. [Sharpening Algorithms](#sharpening-algorithms)
4. [Deconvolution Algorithms](#deconvolution-algorithms)
5. [Algorithm Selection Guide](#algorithm-selection-guide)
6. [Mathematical Background](#mathematical-background)
7. [Performance Comparison](#performance-comparison)

---

## Overview

Reson v0 implements **12 enhancement algorithms** operating in the spatial domain:
- **5 Denoising Algorithms** - Remove noise while preserving structure
- **4 Sharpening Algorithms** - Enhance edges and fine details
- **3 Deconvolution Algorithms** - Reverse optical blur using PSF

All algorithms work on grayscale or RGB images and process in the [0, 1] float range internally.

---

## Denoising Algorithms

### 1. Bilateral Filter

**Class:** `BilateralFilter`  
**Type:** Edge-preserving smoothing  
**Speed:** ⚡⚡⚡ Very Fast  
**Best for:** General-purpose denoising

#### How it Works

The bilateral filter is a non-linear filter that smooths images while preserving edges by considering both:
1. **Spatial proximity** - Pixels close in space
2. **Intensity similarity** - Pixels similar in value

For each pixel, it computes a weighted average of nearby pixels where weights depend on both distance and intensity difference.

#### Parameters

```yaml
type: BilateralFilter
params:
  d: 9                    # Diameter of pixel neighborhood
  sigma_color: 75         # Filter sigma in color space
  sigma_space: 75         # Filter sigma in coordinate space
```

**Parameter Guide:**
- `d` (5-15): Larger = more smoothing, slower
  - 5: Minimal smoothing
  - 9: **Recommended**
  - 15: Heavy smoothing
  
- `sigma_color` (10-150): Controls how much intensity difference is allowed
  - 50: Preserve strong edges only
  - 75: **Balanced**
  - 100-150: Aggressive smoothing
  
- `sigma_space` (10-150): Controls spatial influence
  - 50: Local smoothing
  - 75: **Balanced**
  - 100+: Wider smoothing

#### Pros & Cons

✅ **Advantages:**
- Fast execution
- Excellent edge preservation
- Good for most microscopy images
- Minimal parameter tuning needed

❌ **Disadvantages:**
- Can leave some noise in uniform regions
- May create "cartoon" effect with very high sigma_color
- Less effective on heavy noise

#### Use Cases
- General microscopy images
- Real-time or batch processing
- When edge preservation is critical
- First-pass denoising

---

### 2. Non-Local Means (NLM)

**Class:** `NonLocalMeans`  
**Type:** Patch-based denoising  
**Speed:** 🐌 Slow  
**Best for:** Maximum quality

#### How it Works

Instead of averaging nearby pixels, NLM searches for similar patches (small regions) throughout the image. Each pixel is replaced by a weighted average of pixels from similar patches, even if they're far away.

**Key Insight:** Noise is random, but structure repeats. By finding similar structures elsewhere, we can average out noise while preserving details.

#### Parameters

```yaml
type: NonLocalMeans
params:
  h: 10                   # Filter strength
  template_window_size: 7 # Template patch size
  search_window_size: 21  # Search area size
```

**Parameter Guide:**
- `h` (5-20): Controls denoising strength
  - 5-8: Mild denoising
  - 10: **Recommended**
  - 15+: Heavy denoising (may blur)
  
- `template_window_size` (5-11): Size of patches to compare (must be odd)
  - 5: Small details
  - 7: **Recommended**
  - 9-11: Larger structures
  
- `search_window_size` (15-31): How far to search for similar patches
  - 15: Faster, local search
  - 21: **Balanced**
  - 31: Slower, global search

#### Pros & Cons

✅ **Advantages:**
- Best denoising quality
- Excellent detail preservation
- Works on heavy noise
- Adaptive to image content

❌ **Disadvantages:**
- Very slow (10-20x slower than bilateral)
- Memory intensive
- Can over-smooth with wrong parameters
- Computational complexity O(n²)

#### Use Cases
- Publication-quality images
- Heavily degraded samples
- When processing time is not critical
- Small image sets where quality matters most

---

### 3. Gaussian Denoising

**Class:** `GaussianDenoising`  
**Type:** Linear smoothing  
**Speed:** ⚡⚡⚡⚡ Extremely Fast  
**Best for:** Quick preview, clean images

#### How it Works

Simple Gaussian blur - convolves image with a Gaussian kernel. Each pixel becomes a weighted average of neighbors where weights follow a Gaussian (bell curve) distribution.

**Formula:** G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)

#### Parameters

```yaml
type: GaussianDenoising
params:
  sigma: 1.0              # Standard deviation of Gaussian kernel
```

**Parameter Guide:**
- `sigma` (0.5-3.0): Controls blur strength
  - 0.5-1.0: Mild smoothing
  - 1.0: **Recommended**
  - 1.5-3.0: Heavy blur

#### Pros & Cons

✅ **Advantages:**
- Fastest algorithm
- Predictable behavior
- No artifacts
- Good for low-noise images

❌ **Disadvantages:**
- Blurs edges (not edge-preserving)
- Limited denoising power
- Can make images look "soft"
- Not suitable for heavy noise

#### Use Cases
- Quick preview
- Already clean images needing slight smoothing
- Pre-processing before sharpening
- Real-time applications

---

### 4. Median Filter

**Class:** `MedianFilter`  
**Type:** Order-statistic filter  
**Speed:** ⚡⚡ Fast  
**Best for:** Salt-and-pepper noise

#### How it Works

Replaces each pixel with the median value of neighboring pixels. The median is the middle value when pixels are sorted, making it highly resistant to outliers (noise spikes).

**Example:** Neighbors = [10, 12, 11, 255, 13] → Median = 12 (ignores 255 spike)

#### Parameters

```yaml
type: MedianFilter
params:
  kernel_size: 5          # Size of neighborhood (must be odd)
```

**Parameter Guide:**
- `kernel_size` (3, 5, 7, 9): Window size
  - 3: Minimal smoothing, preserves detail
  - 5: **Recommended**
  - 7-9: Aggressive (may lose detail)

#### Pros & Cons

✅ **Advantages:**
- Excellent for impulse noise (salt-and-pepper)
- Edge-preserving
- Simple and effective
- No parameters to tune (just size)

❌ **Disadvantages:**
- Can create "blocky" artifacts
- Removes fine details at larger kernel sizes
- Not effective on Gaussian noise
- Can alter image statistics

#### Use Cases
- Removing dead/hot pixels
- Salt-and-pepper noise
- CCD/CMOS sensor defects
- Quick noise removal

---

### 5. Anisotropic Diffusion

**Class:** `AnisotropicDiffusion`  
**Type:** PDE-based iterative smoothing  
**Speed:** 🐌 Slow  
**Best for:** Edge preservation with strong smoothing

#### How it Works

Inspired by physical diffusion but "anisotropic" - diffusion rate depends on direction. Smooths strongly within uniform regions but preserves edges by reducing diffusion across high gradients.

**Process:**
1. Calculate gradient at each pixel
2. Diffuse based on gradient magnitude
3. Repeat for multiple iterations
4. Edges stay sharp, uniform regions smooth

#### Parameters

```yaml
type: AnisotropicDiffusion
params:
  num_iterations: 15      # Number of diffusion steps
  kappa: 50               # Edge sensitivity threshold
  gamma: 0.1              # Diffusion rate
```

**Parameter Guide:**
- `num_iterations` (5-50): More = stronger smoothing
  - 5-10: Mild
  - 15: **Recommended**
  - 30-50: Heavy smoothing
  
- `kappa` (20-100): Edge detection threshold
  - 20-30: Preserve weak edges
  - 50: **Balanced**
  - 80-100: Only preserve strong edges
  
- `gamma` (0.05-0.25): Step size per iteration
  - 0.05-0.1: Stable
  - 0.1: **Recommended**
  - 0.2+: Faster but may be unstable

#### Pros & Cons

✅ **Advantages:**
- Excellent edge preservation
- Smooth within regions, sharp at boundaries
- Mathematically well-founded (PDE)
- Highly controllable

❌ **Disadvantages:**
- Slow (iterative process)
- Complex parameter interaction
- Can create "staircase" artifacts
- May sharpen noise if kappa is too high

#### Use Cases
- Segmentation pre-processing
- When edge location is critical
- Medical/scientific imaging
- Combining smoothing with edge detection

---

## Sharpening Algorithms

### 1. Unsharp Masking

**Class:** `UnsharpMasking`  
**Type:** High-pass filtering  
**Speed:** ⚡⚡⚡ Fast  
**Best for:** General sharpening

#### How it Works

Classic sharpening technique:
1. Create blurred version of image
2. Subtract blur from original → "unsharp mask" (high-frequency details)
3. Add amplified mask back to original

**Formula:** Sharp = Original + amount × (Original - Blur)

#### Parameters

```yaml
type: UnsharpMasking
params:
  sigma: 1.0              # Blur radius for mask
  amount: 1.5             # Sharpening strength
  threshold: 0            # Minimum contrast to sharpen
```

**Parameter Guide:**
- `sigma` (0.5-3.0): Scale of features to enhance
  - 0.5-1.0: Fine details
  - 1.0-1.5: **General purpose**
  - 2.0-3.0: Larger structures
  
- `amount` (0.5-3.0): How much to sharpen
  - 0.5-1.0: Subtle
  - 1.5: **Recommended**
  - 2.0-3.0: Aggressive (watch for halos)
  
- `threshold` (0-10): Avoid sharpening low-contrast areas
  - 0: Sharpen everything
  - 1-2: **Reduce noise amplification**
  - 5+: Only sharpen strong edges

#### Pros & Cons

✅ **Advantages:**
- Simple and effective
- Well-understood behavior
- Fast execution
- Works on all image types

❌ **Disadvantages:**
- Can create halos around edges
- Amplifies noise
- Not edge-aware
- May over-sharpen

#### Use Cases
- General microscopy enhancement
- Post-processing after denoising
- Quick sharpening
- When halos are acceptable

---

### 2. Bilateral Sharpening

**Class:** `BilateralSharpening`  
**Type:** Edge-aware sharpening  
**Speed:** ⚡⚡ Moderate  
**Best for:** Halo-free sharpening

#### How it Works

Similar to unsharp masking but uses bilateral filter instead of Gaussian blur:
1. Create edge-preserving blurred version (bilateral filter)
2. Compute detail layer (original - bilateral blur)
3. Add enhanced details back

**Key Advantage:** Bilateral filtering preserves edges, so detail enhancement doesn't create halos.

#### Parameters

```yaml
type: BilateralSharpening
params:
  d: 9
  sigma_color: 75
  sigma_space: 75
  amount: 2.0
```

**Parameter Guide:**
- `d, sigma_color, sigma_space`: Same as BilateralFilter
- `amount` (0.5-3.0): Sharpening strength
  - 1.0: Mild
  - 2.0: **Recommended**
  - 3.0: Aggressive

#### Pros & Cons

✅ **Advantages:**
- No halos around edges
- Strong edge enhancement
- Natural appearance
- Reduces over-sharpening artifacts

❌ **Disadvantages:**
- Slower than unsharp masking
- More parameters to tune
- Can still amplify noise
- May create "cartoon" effect if too strong

#### Use Cases
- Publication-quality images
- When halos are unacceptable
- Processing after bilateral denoising
- High-contrast samples

---

### 3. Guided Filter

**Class:** `GuidedFilter`  
**Type:** Edge-preserving linear filter  
**Speed:** ⚡⚡⚡ Fast  
**Best for:** Smooth gradients, detail enhancement

#### How it Works

A fast edge-preserving filter that avoids gradient reversal (halos). Acts as a "smart" smoothing filter guided by the image structure itself.

**Key Property:** Output gradients never exceed input gradients → no halos by design.

#### Parameters

```yaml
type: GuidedFilter
params:
  radius: 8               # Filter window radius
  eps: 0.01               # Regularization (edge preservation)
  amount: 1.5             # Enhancement strength
```

**Parameter Guide:**
- `radius` (4-16): Smoothing scale
  - 4-6: Fine details
  - 8: **Recommended**
  - 12-16: Larger structures
  
- `eps` (0.001-0.1): Edge preservation
  - 0.001-0.01: Sharp edges (**0.01 recommended**)
  - 0.01-0.1: Smoother gradients
  
- `amount` (0.5-3.0): Sharpening strength

#### Pros & Cons

✅ **Advantages:**
- Mathematically guaranteed no halos
- Fast (linear time complexity)
- Smooth, natural results
- Works well with gradients

❌ **Disadvantages:**
- Less dramatic sharpening than unsharp masking
- Complex parameter interaction
- May not enhance edges as much as desired
- Requires understanding of eps parameter

#### Use Cases
- Images with smooth gradients
- Avoiding overshoot artifacts
- Fast batch processing
- When subtle enhancement is preferred

---

### 4. Laplacian Sharpening

**Class:** `LaplacianSharpening`  
**Type:** Second-derivative edge enhancement  
**Speed:** ⚡⚡⚡⚡ Very Fast  
**Best for:** Quick high-frequency boost

#### How it Works

Uses Laplacian operator (second derivative) to detect edges:
1. Compute Laplacian: ∇²I = ∂²I/∂x² + ∂²I/∂y²
2. Add Laplacian back to original

**Result:** Enhances zero-crossings (edges) and high-frequency content.

#### Parameters

```yaml
type: LaplacianSharpening
params:
  amount: 1.0             # Sharpening strength
```

**Parameter Guide:**
- `amount` (0.5-2.0): Enhancement factor
  - 0.5-1.0: Subtle (**1.0 recommended**)
  - 1.5-2.0: Strong (may amplify noise)

#### Pros & Cons

✅ **Advantages:**
- Extremely fast
- Simple (one parameter)
- Enhances fine details
- Good for isotropic features

❌ **Disadvantages:**
- Very sensitive to noise
- Can create strong artifacts
- Not edge-aware
- No control over scale

#### Use Cases
- Pre-sharpened clean images
- Quick preview
- High-frequency emphasis
- When speed is critical

---

## Deconvolution Algorithms

**New in v1:** Deconvolution algorithms reverse optical blur using Point Spread Function (PSF) modeling.

💡 **See the comprehensive [Deconvolution Guide](06_Deconvolution_Guide.md) for detailed workflow and examples.**

### What is Deconvolution?

Deconvolution reverses image blur caused by the microscope's optical system:

```
Observed Image = True Image ⊗ PSF + Noise
                      ↓ (deconvolution)
Deconvolved Image ≈ True Image
```

Unlike sharpening (which amplifies high frequencies), deconvolution uses knowledge of the PSF to intelligently restore the original image.

**Key Requirements:**
1. **Accurate PSF** - From theory, measurement, or estimation
2. **PSF-limited blur** - Not motion blur or other artifacts
3. **Reasonable SNR** - Very noisy images need pre-denoising

**When to use:** Diffraction-limited imaging, fluorescence microscopy, quantitative analysis

**When NOT to use:** Motion blur, unknown blur, already sharp images

📖 **PSF Generation:** See [PSF Generation Guide](05_PSF_Generation.md) for creating accurate PSFs.

---

### 1. Richardson-Lucy Deconvolution

**Class:** `RichardsonLucy`  
**Type:** Iterative maximum likelihood  
**Speed:** ⚡⚡ Moderate (iterative)  
**Best for:** Fluorescence microscopy, Poisson noise

#### How it Works

Iteratively refines estimate by comparing convolved estimate with observed image:

```
u^(k+1) = u^(k) · [PSF* ⊗ (y / (PSF ⊗ u^(k)))]
```

Based on Bayesian maximum likelihood for Poisson statistics (photon counting).

#### Parameters

```yaml
type: RichardsonLucy
params:
  psf_method: "gibson_lanni"    # PSF generation method
  psf_params:                    # PSF-specific parameters
    wavelength: 520              # nm (emission wavelength)
    numerical_aperture: 1.4
    pixel_size: 0.065            # µm
    size: 31
    ni: 1.518                    # Immersion medium (oil)
    ns: 1.33                     # Sample (aqueous)
    ti: 150                      # Depth (µm)
  iterations: 20                 # 10-50 typical
  regularization: 0.001          # Prevents negative values
```

**PSF Methods:**
- `"gaussian"` - Fast approximation (Rayleigh criterion)
- `"airy"` - Diffraction-limited (Bessel function)
- `"gibson_lanni"` - Fluorescence-specific (most accurate)
- `"custom"` - Load measured PSF from file
- `"blind"` - Estimate PSF from image

**Parameter Guide:**
- `iterations` (10-50): Number of refinement steps
  - 10-15: Mild, safe
  - 20-30: **Balanced (recommended)**
  - 40-50: Aggressive (may over-sharpen)
  
- `regularization` (0.0-0.01): Prevents negative values
  - 0.001: **Light (recommended)**
  - 0.01: Heavy (reduces sharpness slightly)

#### Pros & Cons

✅ **Advantages:**
- Physically motivated (Poisson statistics)
- Excellent for fluorescence
- Preserves non-negativity
- Handles photon noise well
- Most popular deconvolution method

❌ **Disadvantages:**
- Slower (iterative)
- Can over-sharpen with too many iterations
- Requires accurate PSF
- PSF accuracy critical

#### Use Cases
- Widefield fluorescence microscopy
- Photon counting (Poisson noise)
- Resolution enhancement
- Quantitative imaging
- Publication-quality images

**Typical performance:** ~1.5s for 512×512, 31×31 PSF, 20 iterations

---

### 2. Wiener Deconvolution

**Class:** `WienerDeconvolution`  
**Type:** Frequency domain, closed-form  
**Speed:** ⚡⚡⚡⚡ Very Fast  
**Best for:** Quick processing, Gaussian noise

#### How it Works

Solves deconvolution in frequency domain using single FFT operation:

```
U(f) = [H*(f) / (|H(f)|² + K)] · Y(f)
```

Where K is noise-to-signal ratio. Balances deconvolution with noise suppression.

#### Parameters

```yaml
type: WienerDeconvolution
params:
  psf_method: "airy"
  psf_params:
    wavelength: 550
    numerical_aperture: 1.4
    pixel_size: 0.065
    size: 31
  noise_power: 0.01              # Regularization parameter
```

**Parameter Guide:**
- `noise_power` (0.001-0.1): Noise-to-signal ratio
  - 0.001: Low noise, aggressive
  - 0.01: **Moderate (recommended)**
  - 0.1: High noise, conservative

#### Pros & Cons

✅ **Advantages:**
- Extremely fast (~0.1s)
- Closed-form solution (non-iterative)
- Good for Gaussian noise
- Excellent for previews
- Simple (one parameter)

❌ **Disadvantages:**
- Can produce negative values
- Less effective for Poisson noise
- May introduce ringing
- Not as high quality as Richardson-Lucy

#### Use Cases
- Brightfield microscopy
- Fast previews
- High-throughput processing
- Gaussian noise dominant
- When speed is critical

**Typical performance:** ~0.1s for 512×512, 31×31 PSF

---

### 3. Total Variation (TV) Deconvolution

**Class:** `TVDeconvolution`  
**Type:** Regularized optimization  
**Speed:** ⚡ Slow (iterative optimization)  
**Best for:** Edge preservation, noisy images

#### How it Works

Minimizes data fidelity term plus total variation regularization:

```
minimize: ||PSF ⊗ u - y||² + λ · TV(u)
```

Where TV(u) = sum of image gradients. Encourages piecewise smooth solutions.

#### Parameters

```yaml
type: TVDeconvolution
params:
  psf_method: "gaussian"
  psf_params:
    wavelength: 550
    numerical_aperture: 1.0
    pixel_size: 0.1
    size: 31
  iterations: 100                # More than RL
  lambda_tv: 0.01                # Regularization strength
```

**Parameter Guide:**
- `iterations` (50-200): Optimization steps
  - 50: Fast, may not converge
  - 100: **Balanced (recommended)**
  - 200: Careful optimization
  
- `lambda_tv` (0.001-0.1): Edge preservation vs smoothness
  - 0.001: Weak, sharper
  - 0.01: **Moderate (recommended)**
  - 0.1: Strong, may over-smooth

#### Pros & Cons

✅ **Advantages:**
- Excellent edge preservation
- Reduces noise while deconvolving
- Good for piecewise smooth images
- Handles both Gaussian and Poisson noise

❌ **Disadvantages:**
- Slower than other methods
- Can create "staircase" artifacts
- May over-smooth fine details
- More parameters to tune

#### Use Cases
- Low-light/noisy fluorescence
- Images with strong edges
- When noise is problematic
- Edge-preserving reconstruction

**Typical performance:** ~5s for 512×512, 31×31 PSF, 100 iterations

---

### Deconvolution Comparison

| Feature | Richardson-Lucy | Wiener | Total Variation |
|---------|-----------------|--------|-----------------|
| **Speed** | ⚡⚡ | ⚡⚡⚡⚡ | ⚡ |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Best noise type** | Poisson | Gaussian | Both |
| **Iterations** | 10-50 | 1 (direct) | 50-200 |
| **Edge preservation** | Good | Poor | Excellent |
| **Typical use** | Fluorescence | Quick/Preview | Noisy images |

**Choosing an algorithm:**
- **Fluorescence microscopy** → Richardson-Lucy + Gibson-Lanni PSF
- **Brightfield, need speed** → Wiener + Airy PSF
- **Very noisy images** → TV + appropriate PSF
- **General purpose** → Richardson-Lucy + Gaussian PSF

---

## Algorithm Selection Guide

### Decision Tree

```
What is your primary goal?

1. Reverse Optical Blur (Deconvolution)?
   ├─ Fluorescence microscopy → RichardsonLucy + Gibson-Lanni PSF
   ├─ Brightfield, need speed → WienerDeconvolution + Airy PSF
   ├─ Very noisy images → TVDeconvolution
   └─ See Deconvolution Guide for full workflow

2. Remove Noise?
   ├─ Light noise, need speed → BilateralFilter
   ├─ Heavy noise, want quality → NonLocalMeans
   ├─ Salt-and-pepper noise → MedianFilter
   ├─ Need edge preservation → AnisotropicDiffusion
   └─ Just quick smoothing → GaussianDenoising

3. Enhance Edges?
   ├─ General purpose → UnsharpMasking
   ├─ Avoid halos → BilateralSharpening or GuidedFilter
   ├─ Maximum speed → LaplacianSharpening
   └─ Natural gradients → GuidedFilter
```

### Recommended Combinations

#### Combo 1: Balanced (Default)
```yaml
- BilateralFilter (d=9) → UnsharpMasking (amount=1.5)
```
**Use:** General microscopy, good speed/quality

#### Combo 2: Maximum Quality
```yaml
- NonLocalMeans (h=10) → BilateralSharpening (amount=2.0)
```
**Use:** Publication images, small datasets

#### Combo 3: Fast Batch
```yaml
- GaussianDenoising (sigma=1.0) → LaplacianSharpening (amount=1.0)
```
**Use:** Thousands of images, real-time

#### Combo 4: Heavy Noise
```yaml
- NonLocalMeans (h=12) → UnsharpMasking (amount=0.8, threshold=2)
```
**Use:** Very degraded samples

#### Combo 5: Fluorescence Deconvolution
```yaml
- BilateralFilter (d=9) → RichardsonLucy (iterations=20)
```
**Use:** Widefield fluorescence, PSF-limited blur

#### Combo 6: Fast Deconvolution
```yaml
- GaussianDenoising (sigma=0.5) → WienerDeconvolution
```
**Use:** Quick previews, brightfield

---

## Mathematical Background

### Bilateral Filter

**Weight function:**
```
w(i,j,k,l) = exp(-((i-k)² + (j-l)²)/(2σ_space²)) × exp(-(I(i,j) - I(k,l))²/(2σ_color²))
```

Where:
- (i,j) = center pixel
- (k,l) = neighbor pixel
- σ_space = spatial standard deviation
- σ_color = range standard deviation

### Non-Local Means

**Pixel estimate:**
```
NLM(i) = Σ w(i,j) × I(j) / Σ w(i,j)
```

**Weight based on patch similarity:**
```
w(i,j) = exp(-||N(i) - N(j)||²_2 / h²)
```

Where N(i) = neighborhood patch around pixel i

### Unsharp Masking

**Enhancement:**
```
Sharp = I + λ × (I - G_σ * I)
```

Where:
- G_σ = Gaussian kernel with std σ
- λ = amount parameter
- * = convolution

### Anisotropic Diffusion

**PDE evolution:**
```
∂I/∂t = div(c(||∇I||) × ∇I)
```

**Conduction coefficient:**
```
c(||∇I||) = exp(-(||∇I||/κ)²)
```

Where κ = edge threshold

---

## Performance Comparison

### Speed Benchmark (1024×1024 grayscale image)

| Algorithm | Time (s) | Relative |
|-----------|----------|----------|
| GaussianDenoising | 0.002 | 1.0× |
| LaplacianSharpening | 0.003 | 1.5× |
| BilateralFilter | 0.010 | 5.0× |
| MedianFilter | 0.015 | 7.5× |
| UnsharpMasking | 0.018 | 9.0× |
| GuidedFilter | 0.025 | 12.5× |
| BilateralSharpening | 0.030 | 15.0× |
| AnisotropicDiffusion | 0.080 | 40.0× |
| NonLocalMeans | 0.120 | 60.0× |

**System:** CPU-only, single-threaded

### Quality vs Speed Trade-off

```
High Quality ▲
             │
        NLM  │  AnisoDiff
             │
   BilatSharp│  BilatFilter
             │
     Unsharp │  Guided
             │
   Gaussian │  Median
             │
  Laplacian │
             └──────────────► Fast
```

---

## Best Practices

### DO ✅

1. **Denoise before sharpening** - Always reduce noise first
2. **Start conservative** - Use low amounts, increase gradually
3. **Test on samples** - Don't process entire dataset blindly
4. **Match algorithm to noise type** - Median for impulse, NLM for Gaussian
5. **Use threshold in unsharp masking** - Prevents noise amplification

### DON'T ❌

1. **Stack multiple sharpening** - One sharpening step is enough
2. **Use high amount with no denoising** - Amplifies noise
3. **Use NLM without reason** - Too slow for routine processing
4. **Ignore visual artifacts** - Metrics don't catch everything
5. **Over-parameterize** - More parameters ≠ better results

---

## Future Algorithms (v1+)

Planned for future versions:

**v1 - Frequency Domain:**
- Wiener Deconvolution
- Richardson-Lucy Deconvolution
- Blind Deconvolution
- PSF Estimation

**v2 - Multi-Frame:**
- Temporal Denoising
- Frame Averaging
- Super-Resolution

**v3 - Advanced:**
- Deep Learning Enhancement
- Fourier-based SIM Reconstruction

---

**Version:** v0  
**Last Updated:** November 2, 2025  
**Author:** Mehul Yadav
