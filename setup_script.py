"""
Setup script for GBERT-Lite package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gbert-lite",
    version="1.0.0",
    author="Pankaj Mishra, Sneha Nagarajan, V. Venkataramanan, Weiwei Jiang",
    author_email="pankaj.mishra@somaiya.edu",
    description="Lightweight Dual-Transformer Framework for Diffusion-Aware Misinformation Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gbert-lite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "nltk>=3.8.0",
        "spacy>=3.7.0",
        "emoji>=2.8.0",
        "beautifulsoup4>=4.12.0",
        "lxml>=4.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "tqdm>=4.66.0",
        "wandb>=0.15.0",
        "tensorboard>=2.14.0",
        "datasets>=2.14.0",
        "kaggle>=1.5.16",
        "networkx>=3.1",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gbert-lite=main:main",
        ],
    },
)