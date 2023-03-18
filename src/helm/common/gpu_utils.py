import torch


def is_cuda_available() -> bool:
    """Checks if CUDA is available on the machine."""
    return torch.cuda.is_available()


def get_torch_device_name() -> str:
    """Return the device name based on whether CUDA is available."""
    return "cuda" if is_cuda_available() else "cpu"


def get_torch_device() -> torch.device:
    """
    Checks if CUDA is available on the machine and returns PyTorch device.
    """
    return torch.device(get_torch_device_name())
