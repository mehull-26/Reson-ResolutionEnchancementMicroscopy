# Reson – Resolution Enhancement Microscopy

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

## Features (v0)

✅ **9 Enhancement Algorithms**
- 5 denoising methods: Bilateral, NonLocalMeans, Gaussian, Median, Anisotropic
- 4 sharpening methods: UnsharpMasking, BilateralSharpening, GuidedFilter, Laplacian

✅ **Quantitative Metrics**
- Sharpness measures (gradient, Laplacian variance)
- Quality metrics (PSNR, SSIM, MSE) with optional ground truth
- JSON output for batch analysis

✅ **Production Ready**
- Multi-format support (PNG, JPG, TIF, BMP)
- 8-bit and 16-bit output
- Fast processing (~0.2s per image)
- Clean progress bar with verbose mode

✅ **Three Built-in Presets**
- `gentle.yaml` - Minimal enhancement, artifact-free
- `default_v0.yaml` - Balanced processing
- `aggressive.yaml` - Maximum quality for noisy images

---

## Development Roadmap

| Version | Focus | Status | Key Features |
|---------|-------|--------|--------------|
| **v0** | Spatial Enhancement | ✅ Complete | Edge-aware denoising/sharpening, YAML config, quantitative metrics |
| **v1** | PSF Deconvolution | 🔄 Planned | Wiener filtering, Richardson-Lucy, model-based reconstruction |
| **v2** | Multi-Camera Fusion | 🔄 Planned | Multi-channel processing, improved sampling density |
| **v3** | Structured Illumination | 🔄 Future | Super-resolution via structured light patterns |

---

## Documentation

Comprehensive guides in [`docs/`](docs/):

1. **[Installation and Setup](docs/01_Installation_and_Setup.md)** - Get started in 5 minutes
2. **[Configuring Presets](docs/02_Configuring_Presets.md)** - Customize enhancement pipeline
3. **[Interpreting Results](docs/03_Interpreting_Results.md)** - Understand metrics and quality
4. **[Enhancement Algorithms](docs/04_Enhancement_Algorithms.md)** - Technical reference for all algorithms

**Quick Commands:**
```bash
# Use different presets
python main.py -i image.jpg -c configs/presets/gentle.yaml
python main.py -i image.jpg -c configs/presets/aggressive.yaml

# Verbose output for debugging
python main.py -i image.jpg -c configs/default_v0.yaml --verbose

# Batch processing with ground truth
python main.py -i data/input/ -g data/ground_truth/ -c configs/default_v0.yaml
```

See [docs/README.md](docs/README.md) for complete command reference and workflows.

---

## Project Structure

```
Reson/
├── configs/              # YAML configuration files
│   ├── default_v0.yaml   # Balanced preset
│   └── presets/          # gentle.yaml, aggressive.yaml
├── enhancement/          # Enhancement algorithms (9 modules)
│   ├── base.py           # Abstract base class
│   ├── denoising.py      # 5 denoising algorithms
│   └── sharpening.py     # 4 sharpening algorithms
├── metrics/              # Quality metrics (PSNR, SSIM, sharpness)
├── pipeline/             # Processing orchestration
├── utils/                # I/O, preprocessing, visualization
├── docs/                 # Comprehensive documentation (4 guides)
├── main.py               # CLI entry point
└── requirements.txt      # Dependencies
```

For detailed architecture, see [Project Structure](docs/01_Installation_and_Setup.md#project-structure).

---

## Technical Highlights

- **Modular Design:** Pluggable enhancement modules with YAML-based configuration
- **Physics-Driven:** PSF modeling and deconvolution planned for v1+
- **Multi-Format:** Proper 8-bit/16-bit handling for various microscopy formats
- **Quantitative:** Built-in metrics for objective quality assessment
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
