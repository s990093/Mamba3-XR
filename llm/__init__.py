"""
Mamba-3 LLM Package

This package contains the Language Modeling adaptation of Mamba-3.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .mamba3_lm import (
    Mamba3LM,
    create_mamba3_tiny,
    create_mamba3_125m,
    create_mamba3_350m,
)

__all__ = [
    "Mamba3LM",
    "create_mamba3_tiny",
    "create_mamba3_125m",
    "create_mamba3_350m",
]
