import argparse
from scale_utils import get_scale_client

parser = argparse.ArgumentParser()
parser.add_argument("--batch_name", type=str, help="Name of the batch to finalize")
parser.add_argument(
    "--credentials_path", type=str, default="prod_env/credentials.conf", help="Path to the credentials file"
)
args = parser.parse_args()

client = get_scale_client(args.credentials_path)
client.finalize_batch(args.batch_name)
