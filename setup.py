from pathlib import Path
from setuptools import setup, find_packages
import re

description = 'Tools for benchmarking action localization models on the ActLoc dataset.'

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(root / 'actloc_benchmark/__init__.py'), 'r') as f:
    version = re.search(r"__version__ = ['\"]([^'\"]+)['\"]", f.read())
    if version:
        version = version.group(1)
    else:
        raise ValueError("Version not found in __init__.py")
    
with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = [line.strip() for line in f if line.strip()]

setup(
    name='actloc_benchmark',
    version=version,
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=dependencies,
    author='RVP and CVG',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/rvp-group/actloc_benchmark/tree/main',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
