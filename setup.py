#!/usr/bin/env python
"""The setup script."""
from __future__ import annotations

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as reqs_file:
    requirements = [req for req in reqs_file.read().splitlines() if not req.startswith(("#", "-"))]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Xinyang Li",
    author_email="xinyang.li@tum.de",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Python package for simulation of integrated sensing and communications",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords="isac",
    name="isac",
    packages=find_packages(),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/xinyanglii/isac",
    version="1.0.1",
    zip_safe=False,
)
