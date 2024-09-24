# Installation

## Create a virtual environment

It is recommended to install HELM into a virtual environment with Python version >=3.9 to avoid dependency conflicts. HELM requires Python >=3.9. To create, a Python virtual environment and activate it, follow the instructions below.

Using [Virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.9 helm-venv

# Activate the virtual environment.
source helm-venv/bin/activate
```

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
# Create a virtual environment.
# Only run this the first time.
conda create -n crfm-helm python=3.9 pip

# Activate the virtual environment.
conda activate crfm-helm
```

## Install HELM

Within this virtual environment, run:

```
pip install crfm-helm
```

### For HEIM (text-to-image evaluation)

To install the additional dependencies to run HEIM, run:

```
pip install "crfm-helm[heim]"
``` 

Some models (e.g., DALLE-mini/mega) and metrics (`DetectionMetric`) require extra dependencies that are 
not available on PyPI. To install these dependencies, download and run the 
[extra install script](https://github.com/stanford-crfm/helm/blob/main/install-heim-extras.sh):

```
bash install-heim-extras.sh
```
