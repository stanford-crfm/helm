from setuptools import setup, find_packages


def get_requirements(path: str):
    requirements = []
    for line in open(path):
        if not line.startswith('-r'):
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
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires="~=3.8",
    include_package_data=True,
    package_dir={"": "src"},
    install_requires=get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "benchmark-run=benchmark.run:main",
            "benchmark-present=benchmark.presentation.present:main",
            "proxy-server=proxy.server:main",
            "proxy-cli=proxy.cli:main",
        ]
    },
    zip_safe=False,
)
