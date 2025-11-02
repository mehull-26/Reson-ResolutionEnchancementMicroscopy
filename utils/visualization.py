"""Visualization utilities for displaying images and metrics."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union


def plot_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    titles: Tuple[str, str] = ('Original', 'Enhanced'),
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Plot side-by-side comparison of original and enhanced images.

    Parameters
    ----------
    original : np.ndarray
        Original image.
    enhanced : np.ndarray
        Enhanced image.
    titles : Tuple[str, str], optional
        Titles for subplots, by default ('Original', 'Enhanced').
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 6).
    dpi : int, optional
        Figure DPI, by default 100.
    save_path : Optional[str], optional
        Path to save figure, by default None.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Determine if grayscale or color
    cmap = 'gray' if len(original.shape) == 2 else None

    # Plot original
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    # Plot enhanced
    axes[1].imshow(enhanced, cmap=cmap)
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


def plot_metrics(
    metrics: Dict[str, float],
    title: str = 'Image Quality Metrics',
    figsize: Tuple[int, int] = (8, 5),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart of image quality metrics.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values.
    title : str, optional
        Plot title, by default 'Image Quality Metrics'.
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 5).
    dpi : int, optional
        Figure DPI, by default 100.
    save_path : Optional[str], optional
        Path to save figure, by default None.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    names = list(metrics.keys())
    values = list(metrics.values())

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, values, color=colors)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )

    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


def plot_enhancement_pipeline(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple images showing pipeline stages.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images at different pipeline stages.
    titles : List[str]
        Titles for each image.
    figsize : Optional[Tuple[int, int]], optional
        Figure size, auto-calculated if None, by default None.
    dpi : int, optional
        Figure DPI, by default 100.
    save_path : Optional[str], optional
        Path to save figure, by default None.
    """
    n_images = len(images)

    if n_images != len(titles):
        raise ValueError("Number of images and titles must match")

    # Auto-calculate figure size
    if figsize is None:
        figsize = (5 * n_images, 5)

    fig, axes = plt.subplots(1, n_images, figsize=figsize, dpi=dpi)

    # Handle single image case
    if n_images == 1:
        axes = [axes]

    # Determine if grayscale or color
    cmap = 'gray' if len(images[0].shape) == 2 else None

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()


def plot_histogram(
    img: np.ndarray,
    title: str = 'Image Histogram',
    bins: int = 256,
    figsize: Tuple[int, int] = (10, 4),
    dpi: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Plot histogram of image intensities.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    title : str, optional
        Plot title, by default 'Image Histogram'.
    bins : int, optional
        Number of histogram bins, by default 256.
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 4).
    dpi : int, optional
        Figure DPI, by default 100.
    save_path : Optional[str], optional
        Path to save figure, by default None.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if len(img.shape) == 3:  # Color image
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            ax.hist(img[..., i].ravel(), bins=bins, color=color,
                    alpha=0.6, label=color.capitalize())
        ax.legend()
    else:  # Grayscale
        ax.hist(img.ravel(), bins=bins, color='gray', alpha=0.7)

    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)

    plt.show()
