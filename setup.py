from setuptools import setup, find_packages
from setuptools.command.install import install

import os
import sys


def get_requirements(*requirements_file_paths):
    requirements = []
    for requirements_file_path in requirements_file_paths:
        with open(requirements_file_path) as requirements_file:
            for line in requirements_file:
                if line[0:2] != "-r":
                    requirements.append(line.strip())
    return requirements


setup(
    name="crfm-benchmarking",
    version="0.1.0",
    description="Benchmark for language models",
    long_description="Benchmark for language models",
    url="https://github.com/stanford-crfm/benchmarking",
    author="CRFM",
    author_email="contact-crfm@stanford.edu",
    license="Apache License 2.0",
    keywords="language models benchmarking",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
    ],
    #py_modules=["codalab_service"],
    python_requires="~=3.6",
    #cmdclass={"install": Install},
    include_package_data=True,
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "run-benchmark=benchmark.run_benchmark:main",
            "server=benchmark.server:main",
        ]
    },
    zip_safe=False,
)
