import os
from typing import List
import pyhocon


def singleton(items: List):
    if len(items) != 1:
        raise ValueError(f"Expected 1 item, got {len(items)}")
    return items[0]


def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def parse_hocon(text: str):
    return pyhocon.ConfigFactory.parse_string(text)


