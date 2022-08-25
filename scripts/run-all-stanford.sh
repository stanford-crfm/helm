: '
Run RunSpecs in parallel on the Stanford NLP cluster with Slurm.
To bypass the proxy server and run in root mode, append --local.

Usage:

  bash scripts/run-all-stanford.sh --suite <Name of suite> <Any additional CLI arguments for benchmark-present>

  e.g.,
  bash scripts/run-all-stanford.sh --suite v1 --max-eval-instances 1000 --num-threads 8 --priority 2 --local

To kill one of the Slurm jobs:

  squeue -u <>
  scancel <Slurm Job ID>
'

function execute {
   # Prints and executes command
   echo $1
   # eval "time $1"
}

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
  "together/bloom"
  # "together/glm"
  "together/gpt-j-6b"
  "together/gpt-neox-20b"
  "together/opt-66b"
  "together/opt-175b"
  "together/t0pp"
  "together/t5-11b"
  "together/ul2"
  "together/yalm"
)

for model in "${models[@]}"
do
    job="${model//\//-}"  # Replace slashes
    job="${job// /_}"   # Replace spaces

    # Override with passed-in CLI arguments
    # By default, the command will run the RunSpecs listed in src/benchmark/presentation/run_specs.conf
    execute "nlprun --job-name $job --priority high -a crfm_benchmarking -c 4 -g 0 --memory 16g -w /u/scr/nlp/crfm/benchmarking/benchmarking
    'benchmark-present --models-to-run $model $* > $job.log 2>&1'"

    # Run RunSpecs that require a GPU
    execute "nlprun --job-name $job-gpu --priority high -a crfm_benchmarking -c 4 -g 1 --memory 16g -w /u/scr/nlp/crfm/benchmarking/benchmarking
    'benchmark-present --models-to-run $model --conf-path src/benchmark/presentation/run_specs_gpu.conf $* > $job.gpu.log 2>&1'"
done
