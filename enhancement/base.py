"""Base class for enhancement modules."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class EnhancementModule(ABC):
    """
    Abstract base class for all enhancement modules.

    All enhancement modules must inherit from this class and implement
    the `apply` method.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize enhancement module with parameters.

        Parameters
        ----------
        params : Dict[str, Any], optional
            Module-specific parameters, by default None.
        """
        self.params = params or {}
        self.validate_params()

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply enhancement to image.

        Parameters
        ----------
        img : np.ndarray
            Input image (float32, [0, 1] range).

        Returns
        -------
        np.ndarray
            Enhanced image (float32, [0, 1] range).
        """
        pass

    def validate_params(self) -> None:
        """
        Validate module parameters.

        Override this method to add custom parameter validation.
        """
        pass

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Allow module to be called directly.

        Parameters
        ----------
        img : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Enhanced image.
        """
        return self.apply(img)

    def __repr__(self) -> str:
        """String representation of the module."""
        return f"{self.__class__.__name__}(params={self.params})"
