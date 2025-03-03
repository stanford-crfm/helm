import argparse
import os
import subprocess
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(task_id, cli, samples, output_dir):
    output_file = os.path.join(output_dir, f"task_{task_id}_results.json")
    
    cmd = [
        "python", cli,
        "--split", "instruct",
        "--subset", "full",
        "--pass-k", "1",
        "--samples", samples,
        "--selective-evaluate", str(task_id),
        "--output", output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return {"task_id": task_id, "status": "success", "output_file": output_file}
    except subprocess.CalledProcessError as e:
        logger.error(f"Task {task_id} failed: {e}")
        return {"task_id": task_id, "status": "failed", "output_file": None}


def combine_results(output_dir, final_output):
    combined_eval = {}
    all_pass_at_1 = []

    for filename in os.listdir(output_dir):
        if not filename.endswith("_results.json"):
            continue

        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for task_id, task_data in data["results"]["eval"].items():
                    combined_eval[task_id] = task_data

                if "metrics" in data and "pass@1" in data["metrics"]:
                    all_pass_at_1.append(data["metrics"]["pass@1"])
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")

    avg_pass_at_1 = sum(all_pass_at_1) / len(all_pass_at_1) if all_pass_at_1 else 0
    
    final_result = {
        "results": {
            "eval": combined_eval
        },
        "metrics": {
            "split": "instruct",
            "subset": "full",
            "pass@1": avg_pass_at_1,
            "tasks_evaluated": len(all_pass_at_1)
        }
    }
    
    with open(final_output, 'w') as f:
        json.dump(final_result, f, indent=4)
    
    logger.info(f"Combined results from {len(all_pass_at_1)} tasks saved to {final_output}")
    logger.info(f"Average pass@1: {avg_pass_at_1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--samples", default="samples.jsonl")
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--temp-dir", default="temp_results")
    parser.add_argument("--dataset-size", type=int, default=1140)
    
    args = parser.parse_args()
    
    os.makedirs(args.temp_dir, exist_ok=True)

    results = []

    # TODO: replace this to be running each task in a container instead?
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_evaluation, task_id, args.cli, args.samples, args.temp_dir): task_id 
            for task_id in range(1, args.dataset_size + 1)
        }
        
        for future in tqdm(as_completed(futures), total=args.dataset_size, desc="Evaluating tasks"):
            result = future.result()
            results.append(result)

    combine_results(args.temp_dir, args.output)


if __name__ == "__main__":
    main()
