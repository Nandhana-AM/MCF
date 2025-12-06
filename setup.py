"""
Setup script for Amazon Price Prediction package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="MCCF",
    version="1.0.0",
    author="Muthu Palaniappan M",
    author_email="muthupalaniappan.9999@gmail.com",
    description="Dual-backbone deep learning model for product price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<will add later>",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-price-model=train:main",
            "predict-prices=inference:main",
        ],
    },
)