Synthetic PSF-Blurred Test Data
============================================================

This directory contains synthetic test images for deconvolution.

Structure:
  - Each test case has its own folder
  - Each folder contains:
    * blurred.tif - PSF-blurred image (input for deconvolution)
    * ground_truth.tif - Original clean image (for comparison)
    * psf.npy/.tif - PSF used for blurring
    * metadata.txt - Description of test case

Generated 10 test cases

Test with:
  python main.py -i data/synthetic_psf/<case_name>/blurred.tif -g data/synthetic_psf/<case_name>/ground_truth.tif -c configs/deconv_rl.yaml
