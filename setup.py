from setuptools import setup, find_packages

setup(
    name="plant-species-identification",
    version="1.0.0",
    author="Aditya",
    description="A machine learning system for identifying plant species from images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.12.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "flask>=2.3.0",
        "streamlit>=1.25.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
)