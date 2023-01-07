import torch


def get_torch_device() -> torch.device:
    """
    Checks if CUDA is available on the machine and returns PyTorch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
