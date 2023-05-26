from typing import Dict
import argparse
from scaleapi import ScaleClient
import os

parser = argparse.ArgumentParser()
parser.add_argument("--batch_name", type=str, help="Name of the batch to finalize")
args = parser.parse_args()


def get_credentials(path: str) -> Dict[str, str]:
    # Reads the credentials from the given path
    with open(path, "r") as f:
        # Read line by line, replaces the spaces, splits on the first ":"
        # The first part is the key, the second part contians the value in between quotes
        credentials = {}
        for line in f.readlines():
            elt = line.replace(" ", "").replace("\n", "").split(":")
            if len(elt) == 2:
                credentials[elt[0]] = elt[1].split('"')[1]
        return credentials


relative_credentials_path = "../../prod_env/credentials.conf"
credentials_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_credentials_path)
print(f"Reading credentials from {credentials_path}")
credentials = get_credentials(credentials_path)

# Check that scaleApiKey is set
if "scaleApiKey" not in credentials:
    raise Exception("scaleApiKey not found in credentials.conf")

# Get scale client
client = ScaleClient(credentials["scaleApiKey"])
client.finalize_batch(args.batch_name)
