# Contributing to Reson

Thank you for considering contributing to **Reson** - Resolution Enhancement Microscopy! 🔬

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Coding Guidelines](#coding-guidelines)
6. [Submitting Changes](#submitting-changes)
7. [Reporting Bugs](#reporting-bugs)
8. [Suggesting Features](#suggesting-features)
9. [Documentation](#documentation)
10. [Community](#community)

---

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this code:

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Provide helpful feedback and suggestions
- **Be collaborative** - Work together towards common goals
- **Be inclusive** - Welcome newcomers and diverse perspectives

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Found a bug? Help us improve by:

1. **Check existing issues** - Search if the bug is already reported
2. **Create a new issue** with:
   - Clear descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version)
   - Config file used
   - Sample image (if possible)

**Template:**
```
Bug Description:
Brief description of the issue

To Reproduce:
1. Run command: python main.py -i ... -c ...
2. Observe error: ...

Expected Behavior:
What should happen

Environment:
- OS: Windows 11
- Python: 3.10.5
- Reson version: v0

Config File:
[paste your YAML config]

Screenshots/Logs:
[If applicable]
```

### 💡 Suggesting Features

Have an idea? We'd love to hear it!

1. **Check existing issues** - See if it's already proposed
2. **Create a feature request** with:
   - Clear description of the feature
   - Use case / motivation
   - Proposed solution (if you have one)
   - Alternatives considered

**Template:**
```
Feature Request:
Clear description

Motivation:
Why is this needed? What problem does it solve?

Proposed Solution:
How should it work?

Alternatives:
Other approaches considered

Additional Context:
Any other relevant information
```

### 📝 Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation (future)

### 🔧 Contributing Code

Ready to write code? Great! See [Development Setup](#development-setup) below.

---

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Reson-ResolutionEnchancementMicroscope.git
   cd Reson-ResolutionEnchancementMicroscope
   ```

3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope.git
   ```

### Set Up Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install -r requirements-dev.txt  # (to be added)
```

### Verify Installation

```bash
# Run tests (if available)
python -m pytest tests/

# Try processing a sample
python main.py -i data/samples/Z009/Z9_0.jpg -c configs/default_v0.yaml --verbose
```

---

## Project Structure

```
Reson/
├── configs/              # YAML configuration files
│   ├── default_v0.yaml
│   └── presets/
├── enhancement/          # Enhancement algorithms
│   ├── __init__.py
│   ├── base.py          # Abstract base class
│   ├── denoising.py     # Denoising modules
│   └── sharpening.py    # Sharpening modules
├── metrics/              # Quality metrics
│   ├── quality.py       # PSNR, SSIM, MSE
│   └── sharpness.py     # Gradient, Laplacian
├── pipeline/             # Processing orchestration
│   ├── config_loader.py
│   └── processor.py
├── utils/                # Utility functions
│   ├── io.py
│   ├── preprocessing.py
│   └── visualization.py
├── scripts/              # Helper scripts
├── docs/                 # Documentation
├── tests/                # Unit tests (to be added)
├── main.py              # Entry point
└── requirements.txt     # Dependencies
```

### Key Files

- **`enhancement/base.py`** - Base class for all enhancement modules
- **`pipeline/processor.py`** - Main pipeline orchestration
- **`main.py`** - CLI interface

---

## Coding Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use 4 spaces for indentation
def my_function():
    pass

# Maximum line length: 100 characters (not 79)
# Use descriptive variable names
enhanced_image = process(input_image)

# Use type hints
def process(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    pass

# Docstrings in NumPy style
def my_function(param1, param2):
    """
    Brief description.

    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description

    Returns
    -------
    type
        Description
    """
    pass
```

### Code Organization

**For new enhancement modules:**

1. Inherit from `EnhancementModule` base class
2. Implement `apply(image: np.ndarray) -> np.ndarray` method
3. Add to appropriate file (`denoising.py` or `sharpening.py`)
4. Register in `ENHANCEMENT_MODULES` dict in `pipeline/processor.py`

**Example:**

```python
# In enhancement/denoising.py
from .base import EnhancementModule
import numpy as np

class MyNewDenoiser(EnhancementModule):
    """
    Brief description of the algorithm.
    
    Parameters
    ----------
    param1 : type
        Description
    """
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.param1 = self.params.get('param1', default_value)
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image."""
        # Your implementation
        result = your_algorithm(image, self.param1)
        return result

# In pipeline/processor.py
ENHANCEMENT_MODULES = {
    # ... existing modules
    'MyNewDenoiser': MyNewDenoiser,
}
```

### Testing

**Manual testing checklist:**
- [ ] Test on sample images
- [ ] Try different parameter values
- [ ] Check for errors/warnings
- [ ] Verify output quality
- [ ] Test with verbose and non-verbose mode
- [ ] Check JSON metrics output

**Future automated tests (contributions welcome!):**
```python
# tests/test_denoising.py
def test_bilateral_filter():
    """Test BilateralFilter module."""
    from enhancement.denoising import BilateralFilter
    
    # Create test image
    img = np.random.rand(100, 100).astype(np.float32)
    
    # Apply filter
    denoiser = BilateralFilter({'d': 9, 'sigma_color': 75, 'sigma_space': 75})
    result = denoiser.apply(img)
    
    # Assertions
    assert result.shape == img.shape
    assert result.dtype == np.float32
    assert np.all(result >= 0) and np.all(result <= 1)
```

---

## Submitting Changes

### Branch Naming

Use descriptive branch names:
- `feature/your-feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/what-you-changed` - Documentation
- `refactor/what-you-refactored` - Code improvements

### Commit Messages

Write clear commit messages:

```
feat: Add Wiener deconvolution module

- Implement frequency-domain deconvolution
- Add PSF parameter support
- Update documentation

Closes #123
```

**Format:**
- **Type:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- **Subject:** Brief description (50 chars max)
- **Body:** Detailed explanation (optional, wrap at 72 chars)
- **Footer:** Issue references (e.g., "Closes #123")

### Pull Request Process

1. **Update your fork:**
```bash
git checkout main
git pull upstream main
```

2. **Create a branch:**
```bash
git checkout -b feature/my-new-feature
```

3. **Make your changes:**
- Write code
- Add/update tests
- Update documentation
- Test thoroughly

4. **Commit changes:**
```bash
git add .
git commit -m "feat: Add my new feature"
```

5. **Push to your fork:**
```bash
git push origin feature/my-new-feature
```

6. **Open a Pull Request:**
- Go to GitHub
- Click "New Pull Request"
- Fill out the PR template

**PR Template:**
```
Description:
Brief description of changes

Type of Change:
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring

Testing:
- [ ] Tested on sample images
- [ ] Verified metrics output
- [ ] Checked for errors

Screenshots (if applicable):
Before/after comparisons

Checklist:
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Tests pass (if available)
```

### Review Process

1. **Maintainer reviews** your PR
2. **Address feedback** if requested
3. **CI checks** (when available) must pass
4. **Approved** - PR merged!

---

## Reporting Bugs

### Security Vulnerabilities

**DO NOT** open a public issue for security vulnerabilities.

Instead, email: [Add your email]

### Bug Report Guidelines

**Good bug report:**
```
Title: BilateralFilter crashes on 16-bit TIF images

Description:
When processing 16-bit TIFF files with BilateralFilter, 
the program crashes with ValueError.

Steps to Reproduce:
1. Use 16-bit TIF image
2. Run: python main.py -i image.tif -c configs/default_v0.yaml
3. See error

Expected: Should process 16-bit images
Actual: ValueError: array type mismatch

Environment:
- OS: Windows 11
- Python: 3.10.5
- NumPy: 1.24.0

Config:
[paste config]

Stack Trace:
[paste error]
```

---

## Suggesting Features

### Feature Request Guidelines

**Before requesting:**
- Check if feature already exists
- Search existing feature requests
- Consider if it fits project scope

**v0 Scope:** Spatial domain enhancement  
**v1 Scope:** Frequency domain (deconvolution)  
**v2 Scope:** Multi-frame processing  
**v3 Scope:** Advanced methods (SIM, deep learning)

**Good feature request:**
```
Feature: Add Wiener deconvolution

Motivation:
Wiener filtering is standard for deconvolution and would complement 
existing spatial methods. Many users need PSF-based restoration.

Proposed API:
type: WienerDeconvolution
params:
  psf_file: path/to/psf.tif
  noise_power: 0.01

Alternatives:
- Richardson-Lucy (more complex)
- Blind deconvolution (longer-term)

References:
- Paper: [citation]
- Implementation: [link]
```

---

## Documentation

### Documentation Structure

```
docs/
├── README.md                          # Main documentation hub
├── 01_Installation_and_Setup.md      # Getting started
├── 02_Configuring_Presets.md         # Configuration guide
├── 03_Interpreting_Results.md        # Results & metrics
└── 04_Enhancement_Algorithms.md      # Algorithm reference
```

### Writing Documentation

**Guidelines:**
- Use clear, simple language
- Include code examples
- Add screenshots where helpful
- Follow existing structure
- Test all commands/examples

**Example structure:**
```markdown
# Title

Brief introduction

## Section 1

Content...

### Subsection

More specific content...

#### Parameters

- `param1`: Description
- `param2`: Description

#### Example

```yaml
type: AlgorithmName
params:
  param1: value
```

---

## Community

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - Questions, ideas, showcase
- **Pull Requests** - Code contributions

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md (to be created)
- Mentioned in release notes
- Credited in relevant documentation

### Questions?

Don't hesitate to:
- Open an issue with the "question" label
- Start a GitHub Discussion
- Comment on relevant issues/PRs

---

## Development Roadmap

### v0 (Current) - Spatial Enhancement ✅
- [x] Bilateral filtering
- [x] Non-local means
- [x] Unsharp masking
- [x] Multiple algorithms
- [x] YAML configuration
- [x] Comprehensive documentation

### v1 (Planned) - Frequency Domain
- [ ] PSF handling
- [ ] Wiener deconvolution
- [ ] Richardson-Lucy algorithm
- [ ] Blind deconvolution
- [ ] PSF estimation tools

### v2 (Planned) - Multi-Frame
- [ ] Multi-camera fusion
- [ ] Temporal denoising
- [ ] Frame registration
- [ ] Super-resolution

### v3 (Future) - Advanced
- [ ] SIM reconstruction
- [ ] Deep learning models
- [ ] GPU acceleration
- [ ] Real-time processing

### Ongoing
- [ ] Unit tests
- [ ] CI/CD pipeline
- [ ] Performance optimization
- [ ] More documentation
- [ ] Example gallery

---

## Getting Started with Contributing

**First-time contributors:**

1. ⭐ **Star the repository** (helps visibility!)
2. 🍴 **Fork the project**
3. 📥 **Clone your fork**
4. 🛠️ **Set up development environment**
5. 🔍 **Look for "good first issue" labels**
6. 💻 **Make your changes**
7. 🧪 **Test thoroughly**
8. 📤 **Submit a PR**

**Good first issues:**
- Documentation improvements
- Adding examples
- Fixing typos
- Adding docstrings
- Improving error messages
- Adding parameter validation

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (to be specified).

---

## Thank You! 🙏

Every contribution, no matter how small, helps make Reson better for everyone. Whether you're fixing a typo, reporting a bug, or implementing a new feature - thank you for being part of this project!

---

**Questions about contributing?**  
Open an issue with the "question" label or start a GitHub Discussion.

**Ready to contribute?**  
Pick an issue or create a new one and let us know you're working on it!

---

**Project:** Reson - Resolution Enhancement Microscopy  
**Repository:** https://github.com/mehull-26/Reson-ResolutionEnchancementMicroscope  
**Maintainer:** Mehul Yadav  
**Last Updated:** November 2, 2025
