#!/usr/bin/env python3
"""
Setup script for CombustionRL: Reinforcement Learning for Adaptive Solver Selection in Combustion Simulations

This is a research repository accompanying a journal publication.
Not intended for PyPI distribution.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="integrator-rl",
    version="0.1.0",
    author="Eloghosa Ikponmwoba",
    author_email="eloghosaefficiency@gmail.com",
    description="CombustionRL: Reinforcement Learning for Adaptive Solver Selection in Combustion Simulations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/elotech47/CombustionRL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "integrator-rl-train=ppo_training:main",
            "integrator-rl-test=simple_test:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
