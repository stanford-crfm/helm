: '
Run RunSpecs in parallel by models using benchmark-present.

Usage:

  bash scripts/benchmark-present-all-06082022.sh --api-key-path proxy_api_key.4122022.txt --max-eval-instances 1000 --conf-path src/benchmark/presentation/run_specs.conf --num-threads 1

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
execute "benchmark-present --models-to-run openai/davinci openai/code-davinci-001 --dry-run --suite dryrun $*"

models=(
  "ai21/j1-jumbo"
  "ai21/j1-grande"
  "ai21/j1-large"
  # "openai/davinci openai/curie openai/babbage openai/ada"
  # "openai/text-davinci-002 openai/text-davinci-001 openai/text-curie-001 openai/text-babbage-001 openai/text-ada-001"
  # "openai/code-davinci-002 openai/code-davinci-001 openai/code-cushman-001"
  "gooseai/gpt-neo-20b gooseai/gpt-j-6b"
  "anthropic/stanford-online-helpful-v4-s3"
  "microsoft/TNLGv2_530B"
)

for model in "${models[@]}"
do
    suite="${model//\//-}"  # Replace slashes
    suite="${suite// /_}"   # Replace spaces

    # Override with passed-in CLI arguments
    execute "benchmark-present --models-to-run $model --suite 06-08-2022 $* &> $suite.log &"
done
