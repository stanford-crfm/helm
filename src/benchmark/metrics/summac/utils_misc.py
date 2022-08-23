###############################################
# Source: https://github.com/tingofurro/summac
###############################################

import numpy as np
import tqdm
import os
import time

# GPU-related business


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi")
    memory_available = [int(x.split()[2]) + 5 * i for i, x in enumerate(open("tmp_smi", "r").readlines())]
    os.remove("tmp_smi")
    return np.argmax(memory_available)


def any_gpu_with_space(gb_needed):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi")
    memory_available = [float(x.split()[2]) / 1024.0 for i, x in enumerate(open("tmp_smi", "r").readlines())]
    os.remove("tmp_smi")
    return any([mem >= gb_needed for mem in memory_available])


def wait_free_gpu(gb_needed):
    while not any_gpu_with_space(gb_needed):
        time.sleep(30)


def select_freer_gpu():
    freer_gpu = str(get_freer_gpu())
    print("Will use GPU: %s" % (freer_gpu))
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" + freer_gpu
    return freer_gpu


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0:  # Leftovers
        yield batch
