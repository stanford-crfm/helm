# Installation

## Create a virtual environment

It is recommended to install HELM into a virtual environment with Python version >=3.10 to avoid dependency conflicts. HELM requires Python >=3.10. To create, a Python virtual environment and activate it, follow the instructions below.

Using [Virtualenv](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
# Create a virtual environment.
# Only run this the first time.
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.10 helm-venv

# Activate the virtual environment.
source helm-venv/bin/activate
```

Using [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
# Create a virtual environment.
# Only run this the first time.
conda create -n crfm-helm python=3.10 pip

# Activate the virtual environment.
conda activate crfm-helm
```

## Install HELM

Within this virtual environment, run:

```
pip install crfm-helm
```

## Install Multimodal Support

Additional steps are required for multimodal evaluations:

- **HEIM (Text-to-image Model Evaluation)** - to install the additional dependencies to run HEIM (text-to-image evaluation), refer to the [HEIM documentation](heim.md).
- **VHELM (Vision-Language Models)** - To install the additional dependencies to run VHELM (Vision-Language Models), refer to the [VHELM documentation](vhelm.md).
