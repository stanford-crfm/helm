: '
Run RunSpecs in parallel by models using benchmark-present. To bypass the proxy server
and run in root mode, append --local.

Usage:

  bash scripts/benchmark-present-all.sh <Any additional CLI arguments for benchmark-present>

  e.g.,
  bash scripts/benchmark-present-all.sh --max-eval-instances 1000 --num-threads 1 --priority 2 --local

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
  # Not enough OpenAI credits at the moment
  # To conserve OpenAI credits evaluate code-davinci-002 and text-davinci-002
  # instead of code-davinci-001 and text-davinci-001
  # "openai/davinci"
  # "openai/curie"
  # "openai/babbage"
  # "openai/ada"
  # "openai/text-davinci-002"
  "openai/text-curie-001"
  # "openai/text-babbage-001"
  # "openai/text-ada-001"
  # "openai/code-davinci-002"
  "openai/code-cushman-001"
  "gooseai/gpt-j-6b"
  "gooseai/gpt-neo-20b"
  "anthropic/stanford-online-all-v4-s3"
  # "microsoft/TNLGv2_530B"
  # "microsoft/TNLGv2_7B"
  "together/gpt-j-6b"
  "together/gpt-neox-20b"
  "together/opt-66b"
  "together/t0pp"
  "together/t5-11b"
  "together/ul2"
)

for model in "${models[@]}"
do
    logfile="${model//\//-}"  # Replace slashes
    logfile="${logfile// /_}"   # Replace spaces

    # Override with passed-in CLI arguments
    # By default, the command will run the RunSpecs listed in src/benchmark/presentation/run_specs.conf
    # and output results to `benchmark_output/runs/<Today's date e.g., 06-28-2022>`.
    execute "benchmark-present --models-to-run $model $* &> $logfile.log &"
done
