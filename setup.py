# setup.py

from setuptools import setup, find_packages

setup(
    name="border_detection_minimal",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    description="Minimal library for classifying border and core points based on distance threshold.",
    author=":)",
)
