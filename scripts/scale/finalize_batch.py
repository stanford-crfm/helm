import argparse
from scale_utils import get_scale_client

parser = argparse.ArgumentParser()
parser.add_argument("--batch_name", type=str, help="Name of the batch to finalize")
args = parser.parse_args()

client = get_scale_client()
client.finalize_batch(args.batch_name)
