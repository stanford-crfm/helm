: '
Run RunSpecs in parallel by models using benchmark-present.

Usage:

  conda activate crfm_benchmarking
  bash scripts/benchmark-present-all-ai21.sh --api-key-path proxy_api_key.4122022.txt --max-eval-instances 1000 --conf-path src/benchmark/presentation/run_specs.conf --num-threads 2 --priority 2 --local

To kill a running process:

  ps -A | grep benchmark-present
  kill <pid>
'

function execute {
   # Prints and executes command
   echo $1
   eval "time $1"
}

# Perform dry run with just a single model to download and cache all the datasets
# Override with passed-in CLI arguments
# execute "benchmark-present --models-to-run openai/davinci openai/code-davinci-001 --dry-run --suite dryrun $* &> dryrun.log"

models=(
  "ai21/j1-jumbo"
  "ai21/j1-grande"
  "ai21/j1-large"
)

for model in "${models[@]}"
do
    suite="${model//\//-}"  # Replace slashes
    suite="${suite// /_}"   # Replace spaces

    # Override with passed-in CLI arguments
    execute "benchmark-present --models-to-run $model --suite ai21 $* &> $suite.log &"
done
