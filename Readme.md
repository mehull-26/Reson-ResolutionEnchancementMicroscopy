# Reson – Resolution Enhancement Microscopy
**IN DEVELOPMENT (NOT RECOMMENDED TO BE USED BY THE GENERAL PUBLIC, CODE WAS MODIFIED TO OPTIMIZE THE RESULTS, SOME PARTS OF THE CODE MIGHT NOT BE CONSISTENT FOR NOW)**

**Reson** is a computational imaging framework for **resolution enhancement in microscopy** through physics-based reconstruction methods including PSF deconvolution, multi-camera fusion, and structured illumination microscopy (SIM).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-v0%20Complete-success)](https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope)

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope.git
cd Reson-ResolutionEnchancementMicroscope

# Install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Process images
python main.py -i data/samples/Z009 -c configs/default_v0.yaml

# View results
# - Enhanced images: data/processed/default_v0/
# - Metrics (JSON): results/default_v0/
```

📚 **Full Documentation:** [`docs/`](docs/) | **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## Features (v1)

✅ **9 Enhancement Algorithms**
- 5 denoising methods: Bilateral, NonLocalMeans, Gaussian, Median, Anisotropic
- 4 sharpening methods: UnsharpMasking, BilateralSharpening, GuidedFilter, Laplacian

✅ **Deconvolution Algorithms (v1)** 🆕
- Richardson-Lucy: Iterative, optimized for fluorescence/Poisson noise
  * Works with both known PSF parameters and custom measured PSFs

✅ **PSF Generation Methods (v1)** 🆕
- Gaussian: Fast approximation using Rayleigh criterion
- Airy: Diffraction-limited (Bessel function)
- Gibson-Lanni: Fluorescence-specific with aberration modeling
- Custom PSF loading: Load measured PSF from experimental bead imaging
- ⚠️ Blind estimation (experimental): Iterative method, limited accuracy

✅ **Quantitative Metrics**
- Sharpness measures (gradient, Laplacian variance)
- Quality metrics (PSNR, SSIM, MSE) with optional ground truth
- JSON output for batch analysis

✅ **Production Ready**
- Multi-format support (PNG, JPG, TIF, BMP)
- 8-bit and 16-bit output
- Fast processing (~0.2s per image)
- Clean progress bar with verbose mode

✅ **Built-in Presets**
- 3 enhancement presets (gentle, default, aggressive)
- 2 deconvolution presets (RL with known PSF, RL with custom PSF)

---

## Development Roadmap

| Version | Focus | Status | Key Features |
|---------|-------|--------|--------------|
| **v0** | Spatial Enhancement | ✅ Complete | Edge-aware denoising/sharpening, YAML config, quantitative metrics |
| **v1** | PSF Deconvolution | ✅ Complete | Richardson-Lucy, TV deconvolution; 4 PSF methods; Synthetic test data |
| **v2** | Multi-Camera Fusion | 🔄 Planned | Multi-channel processing, improved sampling density |
| **v3** | Structured Illumination | 🔄 Future | Super-resolution via structured light patterns |

---

## Documentation

Comprehensive guides in [`docs/`](docs/):

1. **[Installation and Setup](docs/01_Installation_and_Setup.md)** - Get started in 5 minutes
2. **[Configuring Presets](docs/02_Configuring_Presets.md)** - Customize enhancement pipeline
3. **[Interpreting Results](docs/03_Interpreting_Results.md)** - Understand metrics and quality
4. **[Enhancement Algorithms](docs/04_Enhancement_Algorithms.md)** - Technical reference for all algorithms
5. **[PSF Generation Guide](docs/05_PSF_Generation.md)** 🆕 - Create accurate Point Spread Functions
6. **[Deconvolution Guide](docs/06_Deconvolution_Guide.md)** 🆕 - Complete workflow for image deconvolution

**Quick Commands:**
```bash
# Enhancement (v0)
python main.py -i image.tif -c configs/default_v0.yaml
python main.py -i image.jpg -c configs/presets/gentle.yaml
python main.py -i image.jpg -c configs/presets/aggressive.yaml

# Deconvolution (v1) - Fluorescence microscopy
python main.py -i fluorescence.tif -c configs/deconv_rl.yaml

# Verbose output for debugging
python main.py -i image.jpg -c configs/default_v0.yaml --verbose

# Batch processing with ground truth
python main.py -i data/input/ -g data/ground_truth/ -c configs/default_v0.yaml

# Test with synthetic data
python main.py -i data/synthetic_psf/fluorescent_beads_fluorescence_poisson5/blurred.tif \
    -g data/synthetic_psf/fluorescent_beads_fluorescence_poisson5/ground_truth.tif \
    -c configs/deconv_fluorescence.yaml --verbose
```

See [docs/README.md](docs/README.md) for complete command reference and workflows.

---

## Project Structure

```
Reson/
├── configs/              # YAML configuration files
│   ├── default_v0.yaml   # Balanced enhancement preset
│   ├── deconv_rl_known.yaml     # RL with Gaussian PSF (known parameters)
│   ├── deconv_rl_custom.yaml    # RL with custom measured PSF
│   └── presets/          # gentle.yaml, aggressive.yaml
├── enhancement/          # Enhancement algorithms
│   ├── base.py           # Abstract base class
│   ├── denoising.py      # 5 denoising algorithms
│   ├── sharpening.py     # 4 sharpening algorithms
│   └── deconvolution.py  # Richardson-Lucy deconvolution (v1)
├── utils/                # Utilities
│   ├── psf_generation.py # Theoretical PSF models + custom loading + blind (experimental)
│   ├── io.py             # Image I/O
│   └── visualization.py  # Plotting
├── metrics/              # Quality metrics (PSNR, SSIM, sharpness)
├── pipeline/             # Processing orchestration
├── scripts/              # Utility scripts
│   ├── generate_synthetic_psf.py  # Test data generator with PSF-based blur
│   ├── compare_presets.py         # Results comparison tool
│   └── generate_psf.py            # Standalone PSF generation utility
├── data/
│   └── synthetic_psf/    # 13 test cases with PSF-blurred images (v1)
├── docs/                 # Comprehensive documentation (6 guides)
├── main.py               # CLI entry point
└── requirements.txt      # Dependencies
```

For detailed architecture, see [Project Structure](docs/01_Installation_and_Setup.md#project-structure).

---

**Technical Highlights**

- **Modular Design:** Pluggable enhancement modules with YAML-based configuration
- **Physics-Driven:** PSF-based deconvolution with 4 theoretical models + custom PSF loading
- **Multi-Algorithm:** 10 total algorithms (5 denoising, 4 sharpening, 1 deconvolution)
- **Multi-Format:** Proper 8-bit/16-bit handling for various microscopy formats
- **Quantitative:** Built-in metrics for objective quality assessment (PSNR, SSIM, sharpness)
- **Validated:** Richardson-Lucy deconvolution verified with +8 dB PSNR improvement on synthetic data
- **Tested:** 13 synthetic test cases with ground truth for validation (5 clean + 2 noisy scenarios)
- **Extensible:** Easy to add new algorithms (see [CONTRIBUTING.md](CONTRIBUTING.md))

---

## Vision

**Reson** unifies computational optics and microscopy under one framework — enabling software-based resolution enhancement through **accurate physical modeling**, **multi-channel data fusion**, and **illumination-structured reconstruction** to extend the optical limits of microscopy.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Coding guidelines
- How to add new enhancement algorithms
- Pull request process

**Good first issues:** Documentation improvements, parameter validation, adding tests.

---

## Acknowledgment

If you find Reson useful in your work, please consider:
- ⭐ **Starring this repository** to show your support
- 📝 **Mentioning it in your acknowledgments**:

  > Image enhancement performed using Reson v0 (https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope)

**Note:** Reson is currently in active development (v0 - baseline implementation). Formal citation will be available after peer-reviewed publication of v1 (physics-based methods) or v2 (learning-based methods).

---

**Version:** v0 | **Author:** Mehul Yadav | **Repository:** [GitHub](https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope)
