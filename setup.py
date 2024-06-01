from setuptools import setup, find_packages
import os
import sys
import io


short_description = "LILAC is designed to compare longitudinal images to identify clinically relevant changes."

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, "lilac"))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = short_description

setup(
    name="lilac",
    version="1.0.0",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    author="Heejong Kim",
    author_email="heejong.hj.kim@gmail.com",
    url="https://github.com/heejong-kim/lilac",
    install_requires=[
        "pytorch >= 1.10.1",
        "pandas >= 1.3.4",
        "pillow >= 8.4.0",
        "torchvision >= 0.11.2",
        "numpy >= 1.21.2",
        "tensorboardx >= 2.4.1",
        "torchio >= 0.18.73",
    ],
    license="Apache",
    packages=find_packages(
        exclude=[
            "example_dataset_csv"
        ]
    ),
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)