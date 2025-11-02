# Configuring Presets Guide

## Reson v0 - Creating and Customizing Enhancement Configurations

---

## Table of Contents
1. [Configuration File Structure](#configuration-file-structure)
2. [Built-in Presets](#built-in-presets)
3. [Enhancement Modules](#enhancement-modules)
4. [Creating Custom Presets](#creating-custom-presets)
5. [Parameter Tuning](#parameter-tuning)
6. [Best Practices](#best-practices)

---

## Configuration File Structure

Configuration files are written in YAML format and define the enhancement pipeline.

### Basic Structure

```yaml
# Reson v0 Configuration Template
version: v0
experiment_name: my_experiment  # Output folder name

enhancement:
  modules:
    - name: denoising          # Module name (arbitrary)
      type: BilateralFilter    # Algorithm type
      enabled: true            # Enable/disable module
      params:                  # Algorithm-specific parameters
        d: 9
        sigma_color: 75
        sigma_space: 75

io:
  output_bit_depth: 16         # 8 or 16-bit output

processing:
  clip_output: true            # Clip values to [0,1]

metrics:
  enabled: true
  compute:
    - sharpness
    - laplacian_variance
```

---

## Built-in Presets

### 1. **default_v0.yaml** - Balanced Enhancement

**Best for:** General-purpose microscopy images  
**Speed:** Fast (~0.2s per image)  
**Characteristics:** Moderate noise reduction, gentle sharpening

```yaml
enhancement:
  modules:
    - name: denoising
      type: BilateralFilter
      params:
        d: 9
        sigma_color: 75
        sigma_space: 75
    
    - name: sharpening
      type: UnsharpMasking
      params:
        sigma: 1.0
        amount: 1.5
        threshold: 0
```

**When to use:**
- First-time processing
- Images with moderate noise
- Unknown image quality

### 2. **gentle.yaml** - Minimal Enhancement

**Best for:** Clean images, preventing artifacts  
**Speed:** Very fast (~0.1s per image)  
**Characteristics:** Light denoising, subtle sharpening

```yaml
enhancement:
  modules:
    - name: denoising
      type: GaussianDenoising
      params:
        sigma: 1.0
    
    - name: sharpening
      type: UnsharpMasking
      params:
        sigma: 1.0
        amount: 0.8
```

**When to use:**
- High-quality input images
- Avoiding over-enhancement
- Preserving fine details

### 3. **aggressive.yaml** - Maximum Enhancement

**Best for:** Very noisy/blurry images  
**Speed:** Slow (~0.5s per image)  
**Characteristics:** Strong noise reduction, heavy sharpening

```yaml
enhancement:
  modules:
    - name: denoising
      type: NonLocalMeans
      params:
        h: 10
        template_window_size: 7
        search_window_size: 21
    
    - name: sharpening
      type: BilateralSharpening
      params:
        d: 9
        sigma_color: 75
        sigma_space: 75
        amount: 2.0
```

**When to use:**
- Heavily degraded images
- Synthetic test data
- Maximum quality improvement

**⚠ Warning:** May introduce artifacts on clean images

---

## Enhancement Modules

### Denoising Modules

#### 1. **BilateralFilter** (Recommended for speed)
- **Speed:** ⚡⚡⚡ Very Fast
- **Quality:** ⭐⭐⭐ Good
- **Use case:** General denoising

```yaml
type: BilateralFilter
params:
  d: 9                    # Neighborhood diameter (5-15)
  sigma_color: 75         # Color similarity (50-150)
  sigma_space: 75         # Spatial distance (50-150)
```

**Tuning tips:**
- Increase `d` for more smoothing (slower)
- Increase `sigma_color` for stronger noise reduction
- Increase `sigma_space` for wider smoothing

#### 2. **NonLocalMeans** (Best quality)
- **Speed:** 🐌 Slow
- **Quality:** ⭐⭐⭐⭐⭐ Excellent
- **Use case:** Maximum quality, batch processing

```yaml
type: NonLocalMeans
params:
  h: 10                   # Filter strength (5-15)
  template_window_size: 7 # Template size (5-9)
  search_window_size: 21  # Search area (15-31)
```

**Tuning tips:**
- Increase `h` for more denoising (may blur)
- Reduce `search_window_size` for speed
- Keep `template_window_size` small (7-9)

#### 3. **GaussianDenoising** (Fastest)
- **Speed:** ⚡⚡⚡⚡ Extremely Fast
- **Quality:** ⭐⭐ Fair
- **Use case:** Quick preview, clean images

```yaml
type: GaussianDenoising
params:
  sigma: 1.0              # Blur strength (0.5-3.0)
```

#### 4. **MedianFilter**
- **Speed:** ⚡⚡ Fast
- **Quality:** ⭐⭐⭐ Good for salt-and-pepper noise
- **Use case:** Impulse noise

```yaml
type: MedianFilter
params:
  kernel_size: 5          # Filter size (3, 5, 7)
```

#### 5. **AnisotropicDiffusion**
- **Speed:** 🐌 Slow
- **Quality:** ⭐⭐⭐⭐ Excellent edge preservation
- **Use case:** Preserving edges

```yaml
type: AnisotropicDiffusion
params:
  num_iterations: 15      # Number of iterations (10-50)
  kappa: 50               # Conduction coefficient (30-100)
  gamma: 0.1              # Rate of diffusion (0.05-0.2)
```

### Sharpening Modules

#### 1. **UnsharpMasking** (Standard sharpening)
- **Speed:** ⚡⚡⚡ Fast
- **Quality:** ⭐⭐⭐⭐ Very Good
- **Use case:** General sharpening

```yaml
type: UnsharpMasking
params:
  sigma: 1.0              # Blur radius (0.5-3.0)
  amount: 1.5             # Sharpening strength (0.5-3.0)
  threshold: 0            # Minimum contrast (0-10)
```

**Tuning tips:**
- Increase `amount` for more sharpening
- Increase `sigma` for larger-scale features
- Set `threshold` > 0 to avoid noise amplification

#### 2. **BilateralSharpening** (Edge-aware)
- **Speed:** ⚡⚡ Moderate
- **Quality:** ⭐⭐⭐⭐⭐ Excellent
- **Use case:** Sharp edges without halos

```yaml
type: BilateralSharpening
params:
  d: 9
  sigma_color: 75
  sigma_space: 75
  amount: 2.0
```

#### 3. **GuidedFilter** (Artifact-free)
- **Speed:** ⚡⚡⚡ Fast
- **Quality:** ⭐⭐⭐⭐ Very Good
- **Use case:** Smooth gradients, no halos

```yaml
type: GuidedFilter
params:
  radius: 8               # Filter radius (4-16)
  eps: 0.01               # Regularization (0.001-0.1)
  amount: 1.5
```

#### 4. **LaplacianSharpening** (High-frequency boost)
- **Speed:** ⚡⚡⚡⚡ Very Fast
- **Quality:** ⭐⭐⭐ Good
- **Use case:** Quick sharpening

```yaml
type: LaplacianSharpening
params:
  amount: 1.0             # Sharpening strength (0.5-2.0)
```

---

## Creating Custom Presets

### Example 1: Low-Noise Microscopy

For high-quality microscopy with minimal noise:

```yaml
# configs/custom_clean.yaml
version: v0
experiment_name: clean_microscopy

enhancement:
  modules:
    - name: denoising
      type: GaussianDenoising
      enabled: true
      params:
        sigma: 0.8
    
    - name: sharpening
      type: GuidedFilter
      enabled: true
      params:
        radius: 6
        eps: 0.01
        amount: 1.2

io:
  output_bit_depth: 16

processing:
  clip_output: true

metrics:
  enabled: true
  compute:
    - sharpness
    - laplacian_variance
```

### Example 2: Heavy Noise Reduction

For extremely noisy images:

```yaml
# configs/custom_denoise.yaml
version: v0
experiment_name: heavy_denoise

enhancement:
  modules:
    - name: first_pass
      type: NonLocalMeans
      enabled: true
      params:
        h: 12
        template_window_size: 7
        search_window_size: 21
    
    - name: second_pass
      type: BilateralFilter
      enabled: true
      params:
        d: 7
        sigma_color: 100
        sigma_space: 100
    
    - name: gentle_sharpen
      type: UnsharpMasking
      enabled: true
      params:
        sigma: 1.2
        amount: 0.8
        threshold: 2
```

### Example 3: Fast Batch Processing

For processing thousands of images quickly:

```yaml
# configs/custom_fast.yaml
version: v0
experiment_name: fast_batch

enhancement:
  modules:
    - name: denoising
      type: BilateralFilter
      enabled: true
      params:
        d: 5              # Smaller neighborhood
        sigma_color: 50
        sigma_space: 50
    
    - name: sharpening
      type: LaplacianSharpening
      enabled: true
      params:
        amount: 1.0

io:
  output_bit_depth: 8     # Smaller files

processing:
  clip_output: true

metrics:
  enabled: false          # Skip metrics for speed
```

---

## Parameter Tuning

### General Guidelines

#### Denoising Strength
- **Too weak:** Noise still visible
- **Too strong:** Loss of detail, blurry
- **Sweet spot:** Noise reduced but texture preserved

#### Sharpening Amount
- **Too weak:** No visible improvement
- **Too strong:** Halos, artifacts, noise amplification
- **Sweet spot:** Edges enhanced without artifacts

### Tuning Workflow

1. **Start with default_v0.yaml**
2. **Test on 1-2 sample images**
3. **Adjust one parameter at a time**
4. **Use `--verbose` to see processing details**
5. **Compare before/after visually**
6. **Check metrics in JSON results**

### Testing Parameters

```bash
# Test gentle denoising
python main.py -i sample.jpg -c configs/default_v0.yaml --verbose

# Edit config, then test again
# Repeat until satisfied
```

---

## Best Practices

### ✅ DO

1. **Start conservative** - Use gentle.yaml first
2. **Test on samples** - Don't process entire dataset blindly
3. **Name experiments clearly** - Use descriptive `experiment_name`
4. **Save custom configs** - Keep successful presets
5. **Document changes** - Add comments to YAML files
6. **Compare presets** - Use `scripts/compare_presets.py`

### ❌ DON'T

1. **Over-sharpen** - Avoid `amount` > 3.0
2. **Stack too many modules** - 2-3 modules max
3. **Use NonLocalMeans for real-time** - Too slow
4. **Ignore metrics** - Check JSON results
5. **Process with unknown presets** - Always test first

### Preset Selection Matrix

| Image Type | Noise Level | Recommended Preset | Speed |
|------------|-------------|-------------------|-------|
| High-quality microscopy | Low | `gentle.yaml` | Fast |
| Standard microscopy | Medium | `default_v0.yaml` | Fast |
| Degraded/old samples | High | `aggressive.yaml` | Slow |
| Quick preview | Any | `custom_fast.yaml` | Very Fast |
| Publication quality | Medium | `aggressive.yaml` | Slow |

---

## Command Reference

```bash
# Use default preset
python main.py -i data/samples/ -c configs/default_v0.yaml

# Use gentle preset
python main.py -i data/samples/ -c configs/presets/gentle.yaml

# Use aggressive preset
python main.py -i data/samples/ -c configs/presets/aggressive.yaml

# Use custom preset
python main.py -i data/samples/ -c configs/my_custom.yaml

# Verbose output for tuning
python main.py -i sample.jpg -c configs/default_v0.yaml --verbose

# Compare multiple presets
python scripts/compare_presets.py
```

---

## Next Steps

- **Understand Results**: See [Interpreting Results](03_Interpreting_Results.md)
- **Learn Algorithms**: See [Enhancement Algorithms](04_Enhancement_Algorithms.md)

---

**Version:** v0  
**Last Updated:** November 2, 2025  
**Author:** Mehul Yadav
