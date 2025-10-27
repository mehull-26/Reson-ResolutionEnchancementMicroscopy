
#### In Development
# Reson – Resolution Enhancement Microscopy

**Reson** is a computational imaging framework focused on **resolution enhancement in microscopy**.  
It aims to improve spatial detail and clarity through physics-based reconstruction methods such as PSF deconvolution, multi-camera fusion, and structured illumination microscopy (SIM).  
Real-time performance is pursued wherever feasible but is secondary to reconstruction fidelity.

---

## Goal

To develop a **resolution-enhancing software system** for microscopic imaging, capable of reconstructing higher-quality, higher-resolution representations of observed samples through computational modelling and optical understanding.

---

## Development Roadmap

| Version | Focus | Real-Time Feasibility | Description |
|----------|--------|----------------------|--------------|
| **v0** | Image Enhancement | Real-time | Edge-aware sharpening and denoising to establish data pipeline and base performance. |
| **v1** | PSF-based Deconvolution | Conditional | Uses precomputed or measured PSF for model-based reconstruction; physics-grounded restoration. |
| **v2** | Multi-Camera Fusion | Conditional | Combines multiple imaging channels to improve sampling density and signal-to-noise ratio; integrated mode, not optional. |
| **v3** | Structured Illumination Microscopy (SIM) | Non-real-time | Long-term target for super-resolution via structured light and computational reconstruction; for stable or fluorescence-based imaging setups. |

---

## Technical Highlights

- Physics-driven reconstruction approach (PSF modelling, deconvolution, fusion).  
- Modular design supporting multiple imaging and reconstruction modes.  
- GPU acceleration planned where beneficial; correctness prioritised over speed.  
- Real-time achievable for v0–v2 under suitable computational conditions.  

---

## Vision

**Reson** seeks to unify computational optics and microscopy under one framework —  
enabling software-based resolution enhancement through **accurate physical modeling**, **multi-channel data fusion**, and **illumination-structured reconstruction**.  
Its ultimate aim is to extend the optical limits of microscopy.
