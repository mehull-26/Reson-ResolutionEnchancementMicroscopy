"""Main pipeline processor for image enhancement."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time

from utils.io import load_image, save_image
from utils.preprocessing import to_float, to_uint8, to_uint16
from utils.visualization import plot_comparison, plot_metrics
from enhancement import (
    UnsharpMasking, GuidedFilter, BilateralSharpening, LaplacianSharpening,
    BilateralFilter, NonLocalMeans, GaussianDenoising, MedianFilter, AnisotropicDiffusion
)
from enhancement.deconvolution import RichardsonLucy, WienerDeconvolution, TVDeconvolution
from metrics.quality import psnr, ssim, mse, snr
from metrics.sharpness import (
    gradient_sharpness, laplacian_variance, brenner_sharpness,
    tenengrad_sharpness, variance_sharpness, entropy_sharpness
)


# Mapping of module names to classes
ENHANCEMENT_MODULES = {
    # Sharpening
    'UnsharpMasking': UnsharpMasking,
    'GuidedFilter': GuidedFilter,
    'BilateralSharpening': BilateralSharpening,
    'LaplacianSharpening': LaplacianSharpening,
    # Denoising
    'BilateralFilter': BilateralFilter,
    'NonLocalMeans': NonLocalMeans,
    'GaussianDenoising': GaussianDenoising,
    'MedianFilter': MedianFilter,
    'AnisotropicDiffusion': AnisotropicDiffusion,
    # Deconvolution (PSF-based)
    'RichardsonLucy': RichardsonLucy,
    'WienerDeconvolution': WienerDeconvolution,
    'TVDeconvolution': TVDeconvolution,
}

METRIC_FUNCTIONS = {
    'sharpness': gradient_sharpness,
    'laplacian_variance': laplacian_variance,
    'brenner_sharpness': brenner_sharpness,
    'tenengrad_sharpness': tenengrad_sharpness,
    'variance_sharpness': variance_sharpness,
    'entropy_sharpness': entropy_sharpness,
}


class EnhancementPipeline:
    """
    Main pipeline for orchestrating image enhancement.

    Loads configuration, applies enhancement modules in sequence,
    computes metrics, and saves results.
    """

    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        """
        Initialize enhancement pipeline.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary loaded from YAML.
        verbose : bool
            Whether to show detailed processing information.
        """
        self.config = config
        self.modules = []
        self.original_image = None
        self.enhanced_image = None
        self.metrics_results = {}
        self.verbose = verbose

        # Build enhancement pipeline
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build the enhancement pipeline from configuration."""
        enhancement_config = self.config.get('enhancement', {})
        modules_config = enhancement_config.get('modules', [])

        for module_cfg in modules_config:
            if not module_cfg.get('enabled', True):
                continue

            module_type = module_cfg.get('type')
            module_params = module_cfg.get('params', {})

            if module_type not in ENHANCEMENT_MODULES:
                raise ValueError(f"Unknown enhancement module: {module_type}")

            # Instantiate module
            module_class = ENHANCEMENT_MODULES[module_type]
            module = module_class(params=module_params)

            self.modules.append({
                'name': module_cfg.get('name', module_type),
                'module': module,
                'type': module_type
            })

        if self.verbose:
            print(f"Built pipeline with {len(self.modules)} modules:")
            for i, m in enumerate(self.modules):
                print(f"  {i+1}. {m['name']} ({m['type']})")

    def process(
        self,
        input_path: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        compute_metrics: bool = None,
        visualize: bool = None,
        verbose: bool = None
    ) -> np.ndarray:
        """
        Process an image through the enhancement pipeline.

        Parameters
        ----------
        input_path : Union[str, Path, np.ndarray]
            Path to input image or numpy array.
        output_path : Optional[Union[str, Path]], optional
            Path to save enhanced image, by default None.
        compute_metrics : bool, optional
            Whether to compute metrics, by default None (use config).
        visualize : bool, optional
            Whether to visualize results, by default None (use config).
        verbose : bool, optional
            Whether to show detailed processing information, by default None (use instance setting).

        Returns
        -------
        np.ndarray
            Enhanced image.
        """

        # Use instance verbose if not specified
        if verbose is None:
            verbose = self.verbose

        # Load image
        if isinstance(input_path, np.ndarray):
            self.original_image = to_float(input_path)
        else:
            if verbose:
                print(f"Loading image: {input_path}")
            io_config = self.config.get('io', {})
            self.original_image = load_image(
                input_path,
                as_float=True,
                grayscale=self.config.get('processing', {}).get(
                    'convert_to_grayscale', False)
            )

        if verbose:
            print(
                f"Image shape: {self.original_image.shape}, dtype: {self.original_image.dtype}")

        # Apply enhancement modules
        current_image = self.original_image.copy()

        if verbose:
            print("\nApplying enhancement modules:")
        for i, module_info in enumerate(self.modules):
            start_time = time.time()

            if verbose:
                print(f"  {i+1}. {module_info['name']}...", end=' ')
            current_image = module_info['module'].apply(current_image)

            elapsed = time.time() - start_time
            if verbose:
                print(f"Done ({elapsed:.3f}s)")

        self.enhanced_image = current_image

        # Clip if specified
        if self.config.get('processing', {}).get('clip_output', True):
            self.enhanced_image = np.clip(self.enhanced_image, 0, 1)

        # Compute metrics
        if compute_metrics is None:
            compute_metrics = self.config.get(
                'metrics', {}).get('enabled', False)

        if compute_metrics:
            self._compute_metrics(verbose=verbose)

        # Save image
        if output_path is not None:
            self.save(output_path)

        # Visualize
        if visualize is None:
            visualize = self.config.get('visualization', {}).get(
                'plot_comparison', False)

        if visualize:
            self.visualize()

        return self.enhanced_image

    def _compute_metrics(self, verbose: bool = True) -> None:
        """Compute image quality metrics."""
        if verbose:
            print("\nComputing metrics:")

        metrics_config = self.config.get('metrics', {})
        metric_names = metrics_config.get('compute', [])

        self.metrics_results = {}

        for metric_name in metric_names:
            if metric_name in METRIC_FUNCTIONS:
                metric_fn = METRIC_FUNCTIONS[metric_name]
                value = metric_fn(self.enhanced_image)
                self.metrics_results[metric_name] = value
                if verbose:
                    print(f"  {metric_name}: {value:.4f}")

        # Also compute quality metrics if original is available
        # (This would require a ground truth, so skipping for now)

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save enhanced image to file.

        Parameters
        ----------
        output_path : Union[str, Path]
            Destination file path.
        """
        if self.enhanced_image is None:
            raise ValueError("No enhanced image to save. Run process() first.")

        print(f"\nSaving enhanced image: {output_path}")

        io_config = self.config.get('io', {})
        bit_depth = io_config.get('output_bit_depth', 16)

        save_image(self.enhanced_image, output_path, bit_depth=bit_depth)

    def visualize(self) -> None:
        """Visualize comparison and metrics."""
        if self.original_image is None or self.enhanced_image is None:
            raise ValueError("No images to visualize. Run process() first.")

        print("\nGenerating visualizations...")

        viz_config = self.config.get('visualization', {})
        figsize = tuple(viz_config.get('figsize', [12, 6]))
        dpi = viz_config.get('dpi', 100)

        # Plot comparison
        if viz_config.get('plot_comparison', True):
            plot_comparison(
                self.original_image,
                self.enhanced_image,
                titles=('Original', 'Enhanced'),
                figsize=figsize,
                dpi=dpi
            )

        # Plot metrics
        if viz_config.get('plot_metrics', True) and self.metrics_results:
            plot_metrics(
                self.metrics_results,
                title='Image Quality Metrics',
                figsize=(8, 5),
                dpi=dpi
            )

    def get_metrics(self) -> Dict[str, float]:
        """
        Get computed metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary of metric names and values.
        """
        return self.metrics_results.copy()

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        module_list = ', '.join([m['name'] for m in self.modules])
        return f"EnhancementPipeline(modules=[{module_list}])"
