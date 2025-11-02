"""
Reson - Resolution Enhancement Microscopy
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "Readme.md"
long_description = readme_path.read_text(
    encoding='utf-8') if readme_path.exists() else ""

setup(
    name="reson",
    version="0.1.0",
    author="Mehul",
    description="Resolution Enhancement Microscopy Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "opencv-contrib-python>=4.5.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "PyYAML>=5.4.0",
        "scikit-image>=0.18.0",
        "Pillow>=8.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reson=examples.demo_v0:main",
        ],
    },
)
