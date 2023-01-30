import torch


def get_torch_device_name() -> str:
    """Return the device name based on whether CUDA is avialable."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_torch_device() -> torch.device:
    """
    Checks if CUDA is available on the machine and returns PyTorch device.
    """
    return torch.device(get_torch_device_name())
