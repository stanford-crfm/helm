: '
Run RunSpecs in parallel by models using helm-run.

Usage:

  bash scripts/helm-run-all.sh <Any additional CLI arguments for helm-run>

  e.g.,
  bash scripts/helm-run-all.sh --suite <Name of suite> --max-eval-instances 1000 --num-threads 1 --priority 2

To kill a running process:

  ps -A | grep helm-run
  kill <pid>
'

function execute {
   # Prints and executes command
   echo $1
   eval "time $1"
}

# Perform dry run with just a single model to download and cache all the datasets
# Override with passed-in CLI arguments
# execute "helm-run --models-to-run openai/davinci openai/code-davinci-001 --dry-run --suite dryrun $* &> dryrun.log"

models=(
  "ai21/j1-jumbo"
  "ai21/j1-grande"
  "ai21/j1-large"
  "openai/davinci"
  "openai/curie"
  "openai/babbage"
  "openai/ada"
  "openai/text-davinci-002"
  "openai/text-curie-001"
  "openai/text-babbage-001"
  "openai/text-ada-001"
  "openai/code-davinci-002"
  "openai/code-cushman-001"
  "gooseai/gpt-j-6b"
  "gooseai/gpt-neo-20b"
  "anthropic/stanford-online-all-v4-s3"
  "microsoft/TNLGv2_530B"
  "microsoft/TNLGv2_7B"
  "together/bloom"
  "together/glm"
  "together/gpt-j-6b"
  "together/gpt-neox-20b"
  "together/opt-1.3b"
  "together/opt-6.7b"
  "together/opt-66b"
  "together/opt-175b"
  "together/t0pp"
  "together/t5-11b"
  "together/ul2"
  "together/yalm"
  "cohere/xlarge-20220609"
  "cohere/large-20220720"
  "cohere/medium-20220720"
  "cohere/small-20220720"
)

for model in "${models[@]}"
do
    logfile="${model//\//-}"  # Replace slashes
    logfile="${logfile// /_}"   # Replace spaces

    # Override with passed-in CLI arguments
    # By default, the command will run the RunSpecs listed in src/helm/benchmark/presentation/run_entries.conf
    # and output results to `benchmark_output/runs/<Today's date e.g., 06-28-2022>`.
    execute "helm-run --models-to-run $model $* &> $logfile.log &"
done
