# Reson Documentation

Complete documentation for **Reson** - Resolution Enhancement Microscopy Framework.

---

## Documentation Guide

### 📖 Core Documentation

1. **[Installation and Setup](01_Installation_and_Setup.md)**
   - System requirements
   - Installation steps
   - Project structure
   - Quick start guide
   - Troubleshooting common issues

2. **[Configuring Presets](02_Configuring_Presets.md)**
   - YAML configuration structure
   - All 9 enhancement modules (parameters, ranges, use cases)
   - Built-in presets (gentle, default, aggressive)
   - Creating custom presets
   - Parameter tuning guide

3. **[Interpreting Results](03_Interpreting_Results.md)**
   - Output directory structure
   - Understanding metrics (PSNR, SSIM, sharpness, Laplacian)
   - Reading JSON reports
   - Quality assessment checklist
   - Comparing presets

4. **[Enhancement Algorithms](04_Enhancement_Algorithms.md)**
   - Technical reference for all algorithms
   - Mathematical background
   - Performance benchmarks
   - Algorithm selection guide
   - Best practices

5. **[PSF Generation Guide](05_PSF_Generation.md)** 🆕
   - Point Spread Function theory
   - 4 PSF generation methods (Gaussian, Airy, Gibson-Lanni, Blind)
   - Parameter selection guide
   - Measuring PSF experimentally
   - Troubleshooting PSF issues

6. **[Deconvolution Guide](06_Deconvolution_Guide.md)** 🆕
   - Complete deconvolution workflow
   - 3 algorithms (Richardson-Lucy, Wiener, Total Variation)
   - Algorithm comparison and selection
   - Configuration examples
   - Parameter tuning strategies
   - Troubleshooting artifacts

---

## Quick Reference

### Essential Commands

```bash
# Basic processing
python main.py -i data/samples/Z009 -c configs/default_v0.yaml

# Use different presets
python main.py -i image.jpg -c configs/presets/gentle.yaml
python main.py -i image.jpg -c configs/presets/aggressive.yaml

# Verbose output
python main.py -i image.jpg -c configs/default_v0.yaml --verbose

# With ground truth (for PSNR/SSIM)
python main.py -i data/input/ -g data/ground_truth/ -c configs/default_v0.yaml
```

### Available Presets

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| **gentle.yaml** | ⚡⚡⚡⚡ | ⭐⭐⭐ | Clean images, minimal artifacts |
| **default_v0.yaml** | ⚡⚡⚡ | ⭐⭐⭐⭐ | General-purpose processing |
| **aggressive.yaml** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Noisy images, maximum enhancement |

### Key Metrics

| Metric | Range | Better | Meaning |
|--------|-------|--------|---------|
| **Sharpness** | 0.0-1.0 | Higher | Edge clarity (gradient-based) |
| **Laplacian Variance** | 0-∞ | Higher | Focus quality (second derivative) |
| **PSNR** | 0-∞ dB | Higher | Pixel accuracy vs ground truth |
| **SSIM** | -1 to 1 | Higher | Structural similarity vs ground truth |

*Note: PSNR and SSIM require ground truth images (optional).*

---

## Common Workflows

### Workflow A: First-Time User
1. Read [Installation and Setup](01_Installation_and_Setup.md)
2. Install dependencies and activate environment
3. Test: `python main.py -i data/samples/Z009/Z9_0.jpg -c configs/default_v0.yaml`
4. View results in `data/processed/default_v0/`
5. Process full directory if satisfied

### Workflow B: Parameter Tuning
1. Read [Configuring Presets](02_Configuring_Presets.md)
2. Copy preset: `configs/default_v0.yaml` → `configs/my_preset.yaml`
3. Modify parameters based on guide
4. Test: `python main.py -i sample.jpg -c configs/my_preset.yaml --verbose`
5. Compare metrics and iterate

### Workflow C: Batch Processing
1. Place images in `data/samples/batch1/`
2. Run: `python main.py -i data/samples/batch1 -c configs/default_v0.yaml`
3. Monitor progress bar
4. Review `results/default_v0/overall_report.json`

### Workflow D: Preset Comparison
1. Process with multiple presets (use different experiment names)
2. Compare results visually and via JSON metrics
3. Select best preset for your dataset
4. Apply to full production data

---

## FAQ

**Q: Which preset should I use?**  
**A:** Start with `default_v0.yaml`. Too strong? Use `gentle.yaml`. Too weak? Use `aggressive.yaml`.

**Q: Why are my images white/over-enhanced?**  
**A:** Reduce sharpening `amount` parameter or use `gentle.yaml`. See [Troubleshooting Results](03_Interpreting_Results.md#troubleshooting-poor-results).

**Q: Processing is too slow**  
**A:** Use `BilateralFilter` instead of `NonLocalMeans`, reduce window sizes, or disable metrics. See [Algorithm Selection](04_Enhancement_Algorithms.md#algorithm-selection-guide).

**Q: How do I know if enhancement worked?**  
**A:** Check sharpness metrics (should increase), view images visually, compare PSNR/SSIM if ground truth available. See [Quality Assessment](03_Interpreting_Results.md#quality-assessment-checklist).

**Q: Can I process TIF/16-bit images?**  
**A:** Yes! Supports PNG, TIF (16-bit), JPG (8-bit), BMP. See [Installation](01_Installation_and_Setup.md).

**Q: How do I create a custom preset?**  
**A:** Copy existing config, modify parameters, save as new YAML file. See [Creating Custom Presets](02_Configuring_Presets.md#4-creating-custom-presets).

**Q: What if I get an error?**  
**A:** Run with `--verbose`, check [Troubleshooting](01_Installation_and_Setup.md#troubleshooting), verify environment activated.

---

## Support

**Resources:**
- **Main README:** [../README.md](../README.md) - Project overview
- **Contributing:** [../CONTRIBUTING.md](../CONTRIBUTING.md) - Development guide
- **Examples:** `configs/presets/` - Sample configurations
- **Test Data:** `data/synthetic/` - Sample images with ground truth

**Getting Help:**
1. Check relevant documentation page (see above)
2. Run with `--verbose` flag for detailed output
3. Check [GitHub Issues](https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope/issues)
4. Create new issue with error message, config file, sample image

---

**Version:** v0 | **Last Updated:** November 2, 2025 | [Back to Main README](../README.md)
