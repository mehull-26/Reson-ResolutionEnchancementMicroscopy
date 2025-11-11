# PSF (Point Spread Function) Generation Guide

Complete guide to PSF generation for deconvolution in Reson.

---

## Table of Contents

1. [What is PSF?](#what-is-psf)
2. [Why PSF Matters](#why-psf-matters)
3. [PSF Generation Methods](#psf-generation-methods)
4. [Method Comparison](#method-comparison)
5. [Usage Examples](#usage-examples)
6. [Measuring PSF Experimentally](#measuring-psf-experimentally)
7. [Parameter Selection](#parameter-selection)
8. [Troubleshooting](#troubleshooting)

---

## What is PSF?

### Definition

The **Point Spread Function (PSF)** describes how a microscope images a single point source of light. It represents the blur introduced by the optical system.

**Key concept:** Every point in your sample is blurred by the PSF. The observed image is the convolution of the true sample with the PSF.

```
Observed Image = True Image ⊗ PSF + Noise
```

Where ⊗ denotes convolution.

### Why Images Get Blurred

1. **Diffraction limit** - Light waves interfere, creating a fundamental resolution limit
2. **Aberrations** - Lens imperfections, refractive index mismatch
3. **Defocus** - Sample not exactly at focal plane
4. **Numerical aperture** - Wider apertures collect more light but have larger PSF

---

## Why PSF Matters

### For Deconvolution

Deconvolution algorithms need an accurate PSF to "undo" the blur:

- **Good PSF estimate** → Sharp, accurate deconvolution
- **Poor PSF estimate** → Ringing artifacts, wrong features
- **Wrong PSF** → Can make image worse!

### PSF Characteristics by Microscopy Type

| Microscopy Type | PSF Shape | Key Parameters |
|-----------------|-----------|----------------|
| **Widefield** | Airy disk | NA, wavelength |
| **Confocal** | Narrower Airy | Excitation + emission wavelengths |
| **TIRF** | Thin in z | Evanescent field depth |
| **Light sheet** | Asymmetric | Sheet thickness, NA |

---

## PSF Generation Methods

Reson provides **4 working PSF methods** + 1 experimental:

### Overview Table

| Method | Type | Parameters Needed | Use Case |
|--------|------|-------------------|----------|
| **1. Gaussian** | Theoretical (Known) | λ, NA, pixel size | Quick approximation |
| **2. Airy** | Theoretical (Known) | λ, NA, pixel size | Diffraction-limited |
| **3. Gibson-Lanni** | Theoretical (Known) | λ, NA, ni, ns, ti, pixel size | Fluorescence (best) |
| **4. Custom (Load)** | Measured (Known) | File path | Production use ✅ |
| **5. Blind** | Estimated (Unknown) | Image only | ⚠️ Experimental |

**Key Distinction:**
- **Known PSF (Methods 1-4):** You provide optical parameters → Algorithm calculates/loads PSF
- **Unknown PSF (Method 5):** Algorithm tries to guess PSF from blurred image alone

---

### 1. Gaussian PSF (Theoretical Approximation)

**Best for:** Quick approximation, brightfield, low-NA objectives

```python
from utils.psf_generation import generate_gaussian_psf

psf = generate_gaussian_psf(
    wavelength=550,           # nm (green light)
    numerical_aperture=1.0,   # Objective NA
    pixel_size=0.1,           # µm (camera pixel size)
    size=31                   # PSF size in pixels (odd number)
)
```

**How it works:**
- Approximates PSF as 2D Gaussian
- Width determined by Rayleigh criterion: `σ = 0.21 × λ / NA`
- Simple, fast, reasonably accurate

**Pros:**
- ✅ Very fast
- ✅ Simple parameters
- ✅ Good first approximation

**Cons:**
- ❌ Not physically accurate for high NA
- ❌ Doesn't account for aberrations
- ❌ Underestimates PSF tails

**When to use:**
- Quick tests
- Low NA objectives (< 0.5)
- Brightfield microscopy
- When you don't know exact PSF

---

### 2. Airy PSF (Diffraction-Limited)

**Best for:** More accurate modeling, dry objectives

```python
from utils.psf_generation import generate_airy_psf

psf = generate_airy_psf(
    wavelength=550,
    numerical_aperture=1.4,
    pixel_size=0.065,
    size=31
)
```

**How it works:**
- Uses Airy disk formula: `I(r) = I₀ [2J₁(kr)/(kr)]²`
- `J₁` = first-order Bessel function
- Physically accurate for diffraction-limited imaging

**Pros:**
- ✅ Physically accurate
- ✅ Shows diffraction rings
- ✅ Better for high NA

**Cons:**
- ❌ Slower than Gaussian
- ❌ Still doesn't include aberrations
- ❌ Assumes ideal optics

**When to use:**
- High NA objectives (> 0.9)
- When accuracy matters
- Testing with theoretical limits

---

### 3. Gibson-Lanni PSF (Fluorescence-Specific)

**Best for:** Fluorescence microscopy, accounts for refractive index mismatch

```python
from utils.psf_generation import generate_gibson_lanni_psf

psf = generate_gibson_lanni_psf(
    wavelength=520,           # Emission wavelength (GFP)
    numerical_aperture=1.4,   # Oil immersion objective
    pixel_size=0.065,         # µm
    size=41,                  # Larger for fluorescence
    ni=1.518,                 # Refractive index of oil
    ns=1.33,                  # Refractive index of sample (aqueous)
    ti=150                    # Imaging depth (µm)
)
```

**How it works:**
- Physics-based model for fluorescence
- Accounts for:
  - Refractive index mismatch (coverslip/sample)
  - Spherical aberration
  - Depth-dependent effects

**Pros:**
- ✅ Most accurate for fluorescence
- ✅ Accounts for aberrations
- ✅ Depth-dependent

**Cons:**
- ❌ Slower to compute
- ❌ More parameters needed
- ❌ Simplified version (full 3D is more complex)

**When to use:**
- Fluorescence microscopy
- Oil/water immersion
- Thick samples
- When refractive index mismatch matters

---

### 4. Blind PSF Estimation ⚠️ EXPERIMENTAL

**Status:** ⚠️ **Experimental - Limited Reliability**

**Best for:** Research/exploration when PSF is completely unknown

```python
from utils.psf_generation import estimate_blind_psf

# Estimate PSF from the blurred image itself
psf = estimate_blind_psf(
    image=blurred_image,      # Your blurred image (0-1 range)
    psf_size=31,              # Size of PSF to estimate
    iterations=20             # Number of alternating iterations
)
```

**How it works:**
- Uses iterative alternating optimization (Richardson-Lucy based)
- Alternates between estimating image and PSF
- Requires good initial guess and image content

**⚠️ Important Limitations:**
- **Ill-posed problem**: Single-image blind deconvolution has fundamental mathematical limitations
- **Accuracy varies**: Typically achieves only 50-60% correlation with true PSF
- **Image-dependent**: Success heavily depends on image content and structure
- **Not recommended for production**: Results are often unreliable

**Pros:**
- ✅ No optical parameters needed
- ✅ Can adapt to actual system
- ✅ Useful for PSF exploration/comparison

**Cons:**
- ❌ **Poor accuracy** (~56% correlation with true PSF in testing)
- ❌ Highly ill-posed problem with multiple solutions
- ❌ Sensitive to noise and image content
- ❌ May produce wrong PSF that makes deconvolution worse
- ❌ Computationally expensive

**When to use:**
- ⚠️ Only as last resort when no other option available
- For research/comparison purposes
- To validate theoretical PSF assumptions
- **Not recommended for quantitative work**

**Better alternatives:**
1. **Measure PSF from beads** (best option)
2. **Use theoretical PSF** (Gaussian/Airy/Gibson-Lanni)
3. **Load custom PSF** from manufacturer specs

---

### 5. Custom PSF (Measured Experimentally)

**Best for:** Best accuracy, measured from your actual microscope

```python
from utils.psf_generation import load_custom_psf

# Load measured PSF from file
psf = load_custom_psf(
    psf_file='measured_psf.tif',  # Your measured PSF
    normalize=True
)
```

**Supports:**
- `.tif` / `.tiff` files
- `.npy` files (NumPy arrays)
- `.npz` files (compressed)
- `.png` files

**Requirements:**
- 2D array (if 3D, central slice used)
- Odd dimensions (auto-cropped if even)
- Non-negative values

**When to use:**
- Best possible accuracy needed
- Have access to sub-resolution beads
- Production/publication quality
- System-specific effects important

---

## Method Comparison

### Quick Comparison Table

| Method | Speed | Accuracy | Parameters | Use Case |
|--------|-------|----------|------------|----------|
| **Gaussian** | ⚡⚡⚡⚡ | ⭐⭐ | 3 | Quick test, low NA |
| **Airy** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 3 | High NA, accurate |
| **Gibson-Lanni** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 6 | Fluorescence, best theoretical |
| **Blind (⚠️ Experimental)** | ⚡ | ⭐ | 2 | Research only, not production |
| **Custom (measured)** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 1 | **Production use (best)** |

**Accuracy notes:**
- Custom (measured PSF): Best possible accuracy - reflects actual system
- Gibson-Lanni: Best theoretical model for fluorescence
- Airy: Accurate for diffraction-limited systems
- Gaussian: Reasonable approximation
- **Blind: Poor accuracy (~56% correlation), experimental only**

### Decision Tree

```
Do you have measured PSF from beads?
├─ YES → ✅ Use Custom PSF (best accuracy - RECOMMENDED)
└─ NO
   ├─ Fluorescence microscopy?
   │  ├─ YES → Use Gibson-Lanni PSF (best theoretical model)
   │  └─ NO → Brightfield/phase contrast
   │     ├─ High NA (>0.9)? → Use Airy PSF
   │     └─ Low NA (<0.9)? → Use Gaussian PSF
   └─ Completely unknown system? 
      → ⚠️ Try Blind estimation (experimental, limited accuracy)
      → Better: measure PSF or use theoretical model
```

---

## Usage Examples

### Example 1: Generating PSF for Config File

```bash
# Generate PSF and visualize
python -m scripts.generate_psf \
    --method gibson_lanni \
    --wavelength 520 \
    --na 1.4 \
    --pixel-size 0.065 \
    --output my_psf.tif \
    --visualize
```

Then use in config:
```yaml
psf_method: "custom"
psf_params:
  psf_file: "my_psf.tif"
```

### Example 2: Python API

```python
from utils.psf_generation import get_psf, visualize_psf

# Method 1: Unified interface
psf_params = {
    'wavelength': 520,
    'numerical_aperture': 1.4,
    'pixel_size': 0.065,
    'size': 31,
    'ni': 1.518,
    'ns': 1.33,
    'ti': 150
}

psf = get_psf('gibson_lanni', psf_params)

# Visualize PSF
visualize_psf(psf, title="Gibson-Lanni PSF for GFP")
```

### Example 3: Convenience Functions

```python
from utils.psf_generation import (
    psf_widefield_fluorescence,
    psf_brightfield
)

# Quick fluorescence PSF (oil immersion, GFP)
psf_fluorescence = psf_widefield_fluorescence(
    emission_wavelength=520,
    objective_na=1.4,
    pixel_size=0.065
)

# Quick brightfield PSF
psf_brightfield = psf_brightfield(
    wavelength=550,
    objective_na=0.75,
    pixel_size=0.1
)
```

---

## Measuring PSF Experimentally

### Why Measure PSF?

- **Most accurate** representation of your system
- Captures all aberrations
- System-specific (your microscope, your settings)
- Required for quantitative work

### What You Need

1. **Sub-resolution fluorescent beads**
   - Diameter: 100-200 nm (smaller than diffraction limit)
   - Bright, photostable fluorophore
   - Common: TetraSpeck beads, FluoSpheres

2. **Same imaging conditions**
   - Same objective
   - Same immersion medium
   - Same camera settings
   - Same fluorophore wavelength

### Procedure

1. **Prepare bead sample**
   - Dilute beads (sparse, isolated beads)
   - Mount on coverslip with immersion medium
   - Seal to prevent drift

2. **Image beads**
   - Focus carefully (maximize intensity)
   - Use same settings as real experiments
   - Image multiple beads (10-50)

3. **Extract PSF**
   - Find isolated, well-focused beads
   - Crop region around each bead (e.g., 64×64 pixels)
   - Average multiple beads to reduce noise
   - Normalize to sum = 1

4. **Save PSF**
   ```python
   import numpy as np
   from utils.io import save_image
   
   # Assuming psf_average is your averaged PSF
   psf_normalized = psf_average / psf_average.sum()
   save_image(psf_normalized, 'measured_psf.tif', bit_depth=16)
   np.save('measured_psf.npy', psf_normalized)
   ```

### Quality Checks

✅ **Good PSF:**
- Symmetric (no drift during acquisition)
- Single peak at center
- Smooth falloff
- Sum ≈ 1.0 after normalization

❌ **Bad PSF:**
- Elongated/asymmetric (drift, aberration)
- Multiple peaks (multiple beads)
- Clipped values (overexposure)
- Very noisy (too few beads averaged)

---

## Parameter Selection

### Wavelength (λ)

**For fluorescence:** Use **emission wavelength**, not excitation

| Fluorophore | Excitation | **Emission (use this)** |
|-------------|------------|-------------------------|
| DAPI | 350 nm | **450 nm** |
| GFP | 488 nm | **520 nm** |
| YFP | 514 nm | **530 nm** |
| mCherry/RFP | 587 nm | **610 nm** |
| Cy5 | 649 nm | **670 nm** |

**For brightfield:** Use **illumination wavelength**
- White light: ~550 nm (green, middle of visible spectrum)
- LED: Check manufacturer specs

### Numerical Aperture (NA)

Check objective label:
- **40×/0.95** → NA = 0.95 (dry)
- **60×/1.40 Oil** → NA = 1.40 (oil immersion)
- **63×/1.20 W** → NA = 1.20 (water immersion)

Higher NA → Better resolution → Smaller PSF

### Pixel Size

**Effective pixel size** = Physical pixel size / Magnification

Example:
- Camera: 6.5 µm physical pixels
- Objective: 100×
- **Effective pixel size = 6.5 / 100 = 0.065 µm**

Common values:
- 100× objective + typical sCMOS: **0.065 µm**
- 60× objective: **0.108 µm**
- 40× objective: **0.163 µm**

**Nyquist sampling:** Pixel size should be ≤ λ/(4×NA)
- For λ=520nm, NA=1.4: ≤ 93 nm
- So 0.065 µm (65 nm) is well-sampled ✓

### PSF Size

**Rule of thumb:** PSF should fit comfortably in the kernel

- **Small blur** (low NA, short wavelength): 21×21 pixels
- **Medium blur** (typical): 31×31 pixels
- **Large blur** (high NA, long wavelength, aberrations): 41-51 pixels

**Always use odd numbers** for symmetry!

### Gibson-Lanni Specific Parameters

**Refractive Index (ni) - Immersion Medium:**
- **Air:** 1.0
- **Water:** 1.33
- **Glycerol:** 1.47
- **Oil:** 1.518

**Refractive Index (ns) - Sample:**
- **Aqueous (typical cells):** 1.33
- **Mounting medium:** Check manufacturer
- **Glass:** 1.52

**Working Distance (ti):**
- Distance from coverslip into sample (µm)
- **Surface imaging:** 0-10 µm
- **Typical:** 50-150 µm
- **Deep imaging:** > 200 µm

**Tip:** Larger mismatch (|ni - ns|) → More aberration

---

## Troubleshooting

### Problem: "PSF sums to zero"

**Cause:** Invalid PSF (all zeros or negative values)

**Solution:**
- Check wavelength in nm, not µm
- Check pixel_size in µm, not nm
- Ensure size > 0

### Problem: "PSF too large, slow deconvolution"

**Cause:** PSF size too big

**Solution:**
- Reduce `size` parameter (try 31 instead of 51)
- Check if pixel_size is correct (should be µm, not nm)
- For Gaussian/Airy: Check NA is reasonable (<2.0)

### Problem: "Deconvolution has ringing artifacts"

**Cause:** PSF might be inaccurate

**Solution:**
1. Try different theoretical PSF method (e.g., Gibson-Lanni instead of Gaussian)
2. **Measure PSF experimentally from beads (recommended)**
3. Check parameters are correct (especially wavelength, NA)
4. ⚠️ Blind PSF estimation is experimental and may not help (limited accuracy)

### Problem: "PSF looks wrong when visualized"

**Symptoms:**
- Not centered
- Multiple peaks
- Asymmetric

**Solutions:**
- For generated PSF: Check parameters (especially NA < 2.0)
- For measured PSF: 
  - Re-image with better focus
  - Average more beads
  - Check for drift during acquisition

### Problem: "Blind PSF estimation fails"

**⚠️ This is expected behavior - blind PSF estimation is experimental**

**Why it fails:**
- Single-image blind deconvolution is fundamentally ill-posed
- Multiple PSF/image pairs can produce the same blurred result
- Typically achieves only 50-60% correlation with true PSF
- Highly dependent on image content

**Solution:**
1. ✅ **Measure PSF from fluorescent beads** (best option)
2. ✅ Use theoretical PSF (Gaussian, Airy, Gibson-Lanni)
3. ✅ Load custom PSF from manufacturer specs
4. ⚠️ Accept that blind estimation has limited accuracy
5. Use blind results only for research/exploration, not production

**Note:** Blind PSF estimation is marked as experimental for a reason - it's unreliable for quantitative work.

### Problem: "Custom PSF file not loading"

**Error:** "argument should be a str... not 'ndarray'"

**Solution:**
- Check file path is correct
- Ensure file exists
- For config files: Use relative path from project root

---

## Best Practices

### ✅ DO:
- **Use measured PSF for quantitative work** (best practice)
- Check PSF visualization before using
- Use emission wavelength for fluorescence
- Ensure Nyquist sampling (pixel size ≤ λ/(4×NA))
- Average multiple beads when measuring PSF
- Save PSF in multiple formats (.npy and .tif)
- Verify deconvolution results with metrics (PSNR, SSIM)

### ❌ DON'T:
- Use excitation wavelength (use emission!)
- Mix up units (wavelength in nm, pixel_size in µm)
- Use even-sized PSF (always odd: 21, 31, 41...)
- Measure PSF with overexposed beads
- Use PSF from different microscope/settings
- **Rely on blind PSF estimation for production/quantitative work**
- Assume blind estimation will work reliably

---

## See Also

- **[Deconvolution Guide](06_Deconvolution_Guide.md)** - Using PSF for deconvolution
- **[Enhancement Algorithms](04_Enhancement_Algorithms.md)** - All algorithms reference
- **[Interpreting Results](03_Interpreting_Results.md)** - Evaluating deconvolution quality

---

**Next:** [Deconvolution Guide](06_Deconvolution_Guide.md) - Learn how to use these PSFs for image restoration
