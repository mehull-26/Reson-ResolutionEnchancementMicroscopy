# Installation and Setup Guide

## Reson v0 - Resolution Enhancement Microscopy

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **RAM**: 4 GB (8 GB recommended for large images)
- **Storage**: 500 MB for installation + space for processed images

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16 GB for processing large batches
- **GPU**: Not required (CPU-only implementation)

---

## Installation

### Step 1: Clone or Download the Repository

```bash
git clone https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope.git
cd Reson-ResolutionEnchancementMicroscope
```

Or download and extract the ZIP file.

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Array operations
- `opencv-python` - Image I/O and processing
- `scipy` - Scientific computing
- `scikit-image` - Advanced image processing
- `matplotlib` - Visualization
- `PyYAML` - Configuration file parsing
- `pandas` - Data analysis (optional)

### Step 4: Verify Installation

Test with a single image:

```bash
python main.py -i data/samples/Z009/Z9_0.jpg -c configs/default_v0.yaml
```

Expected output:
```
======================================================================
RESON v0 - Resolution Enhancement Microscopy
======================================================================

Directory: Z9_0.jpg
Pipeline:  denoising → sharpening
Images:    1 files

Progress: |████████████████████████████████████████| [1/1] Complete
Time elapsed: 0.2s

======================================================================
✓ Processing complete!
======================================================================
Enhanced images: B:\...\data\processed\default_v0
Results saved:   B:\...\results\default_v0
======================================================================
```

---

## Project Structure

```
Reson/
├── configs/               # Configuration files
│   ├── default_v0.yaml   # Default preset (balanced)
│   └── presets/
│       ├── gentle.yaml   # Minimal enhancement
│       └── aggressive.yaml # Strong enhancement
│
├── data/
│   ├── samples/          # Your input images (put images here)
│   ├── processed/        # Enhanced images (auto-generated)
│   └── synthetic/        # Test data with ground truth
│
├── results/              # JSON metrics and reports
│   └── [experiment_name]/
│
├── enhancement/          # Enhancement algorithms
│   ├── denoising.py     # Noise reduction modules
│   └── sharpening.py    # Edge enhancement modules
│
├── metrics/              # Quality assessment
│   ├── quality.py       # PSNR, SSIM, MSE
│   └── sharpness.py     # Gradient, Laplacian measures
│
├── pipeline/             # Processing orchestration
│   ├── config_loader.py
│   └── processor.py
│
├── utils/                # Utility functions
│   ├── io.py            # Image loading/saving
│   ├── preprocessing.py
│   └── visualization.py
│
├── scripts/              # Helper scripts
│   ├── generate_synthetic_data.py
│   └── compare_presets.py
│
├── docs/                 # Documentation (you are here!)
│
├── main.py              # Main entry point
└── requirements.txt     # Python dependencies
```

---

## Quick Start

### Basic Usage

**Process a directory of images:**
```bash
python main.py -i data/samples/Z009 -c configs/default_v0.yaml
```

**Process a single image:**
```bash
python main.py -i path/to/image.jpg -c configs/default_v0.yaml
```

**Use a different preset:**
```bash
# Gentle enhancement (minimal artifacts)
python main.py -i data/samples/Z009 -c configs/presets/gentle.yaml

# Aggressive enhancement (maximum sharpening)
python main.py -i data/samples/Z009 -c configs/presets/aggressive.yaml
```

**Verbose output (detailed processing info):**
```bash
python main.py -i data/samples/Z009 -c configs/default_v0.yaml --verbose
```

### Output Locations

After processing, find your results:

- **Enhanced images**: `data/processed/[experiment_name]/`
- **Metrics (JSON)**: `results/[experiment_name]/`
  - `overall_report.json` - Summary statistics
  - `[image]_result.json` - Per-image metrics

---

## Troubleshooting

### Common Issues

#### 1. **Module Not Found Error**
```
ModuleNotFoundError: No module named 'cv2'
```
**Solution:** Activate virtual environment and reinstall dependencies
```bash
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

#### 2. **Out of Memory Error**
**Solution:** Process images in smaller batches or reduce image size

#### 3. **Images Turn White/Black**
**Cause:** Over-enhancement with aggressive settings
**Solution:** Use `gentle.yaml` preset or reduce `amount` parameter in config

#### 4. **Permission Denied**
**Cause:** Output directories are write-protected
**Solution:** Run with appropriate permissions or change output location

#### 5. **Slow Processing**
**Expected:** ~0.2-0.5s per image with default settings
- BilateralFilter: Fast (~0.01s)
- NonLocalMeans: Slow (~0.1s, but better quality)

**Optimization:**
- Use `BilateralFilter` instead of `NonLocalMeans` for speed
- Reduce `search_window_size` in NonLocalMeans
- Process in batches

### Getting Help

1. Check this documentation in `docs/`
2. Review configuration examples in `configs/presets/`
3. Run with `--verbose` flag to see detailed processing info
4. Open an issue on GitHub with error messages and config file

---

## Next Steps

- **Configure Presets**: See [Configuring Presets](02_Configuring_Presets.md)
- **Interpret Results**: See [Interpreting Results](03_Interpreting_Results.md)
- Learn about [Enhancement Algorithms](04_Enhancement_Algorithms.md)

---

**Version:** v0  
**Last Updated:** November 2, 2025  
**Author:** Mehul Yadav
