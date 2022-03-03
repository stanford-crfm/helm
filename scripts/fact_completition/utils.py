"""Assortment of useful utility functions 
"""

import os
import ujson as json
import multiprocessing as mp

def jsonl_generator(fname):
    """ Returns generator for jsonl file """
    for line in open(fname, 'r'):
        line = line.strip()
        if len(line) < 3:
            d = {}
        elif line[len(line)-1] == ',':
            d= json.loads(line[:len(line)-1])
        else:
            d= json.loads(line)
        yield d

def batch_line_generator(fname, batch_size):
    """ Returns generator for jsonl file with batched lines """
    res = []
    batch_id = 0
    for line in open(fname, 'r'):
        line = line.strip()
        if len(line) < 3:
            d = ''
        elif line[len(line) - 1] == ',':
            d = line[:len(line) - 1]
        else:
            d = line
        res.append(d)
        if len(res) >= batch_size:
            yield batch_id, res
            batch_id += 1
            res = []
    yield batch_id, res

def append_to_jsonl_file(data, file):
    """ Appends json dictionary as new line to file """
    with open(file, 'a+') as out_file:
        for x in data:
            out_file.write(json.dumps(x, ensure_ascii=False)+"\n")


def get_batch_files(fdir):
    """ Returns paths to files in fdir """ 
    filenames = os.listdir(fdir)
    filenames = [os.path.join(fdir, f) for f in filenames]
    print(f"Fetched {len(filenames)} files from {fdir}")
    return filenames

def create_dir(out_dir):
    """ Creates new directory if it doesn't already exist """
    if not os.path.exists(out_dir):
        print(f"Creating {out_dir}")
        os.makedirs(out_dir)