: "
Run RunSpecs in parallel on the Stanford NLP cluster with Slurm.
To bypass the proxy server and run in root mode, append --local.

Usage:

  bash scripts/run-all-stanford.sh <Suite name>

To kill one of the Slurm jobs:

  squeue -u $USER
  scancel <Slurm Job ID>
"

function execute {
   # Prints and executes command
   echo $1
   # eval "time $1"
}

cpus=4
num_threads=8
work_dir="/u/scr/nlp/crfm/benchmarking/benchmarking"

models=(
  "ai21/j1-jumbo"
  "ai21/j1-grande"
  "ai21/j1-large"
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
all_models=""
log_paths=()

for model in "${models[@]}"
do
    all_models="${all_models} ${model}"  # Builds a string like ai21/j1-jumbo ai21/j1-grande ai21/j1-large ...
    job="$1-${model//\//-}"  # Replace slashes and prepend suite nam  e.g., openai/curie => <Suite name>-openai-curie

    # Override with passed-in CLI arguments
    # By default, the command will run the RunSpecs listed in src/benchmark/presentation/run_specs.conf
    log_file=$job.log
    execute "nlprun --job-name $job --priority high -a crfm_benchmarking -c $cpus -g 0 --memory 16g -w $work_dir
    'benchmark-present --max-eval-instances 1000 -n $num_threads --priority 2 --local --models-to-run $model --suite $1 > $log_file 2>&1'"
    log_paths+=("$work_dir/$log_file")

    # Run RunSpecs that require a GPU
    log_file=$job.gpu.log
    execute "nlprun --job-name $job-gpu --priority high -a crfm_benchmarking -c $cpus -g 1 --memory 16g -w $work_dir
    'benchmark-present --max-eval-instances 1000 -n $num_threads --priority 2 --local
    --models-to-run $model --conf-path src/benchmark/presentation/run_specs_gpu.conf --suite $1 > $log_file 2>&1'"
    log_paths+=("$work_dir/$log_file")
done

printf "\nTo monitor the runs:\n"
for log_path in "${log_paths[@]}"
do
  echo "tail -f $log_path"
done

# Print out what to run next
printf "\nRun the following commands once the runs complete:\n"
command="benchmark-present --max-eval-instances 1000 --ski-instance --priority 2 --local --models-to-run $all_models --suite $1"
echo "nlprun --job-name generate-run-specs-json-$1 --priority high -a crfm_benchmarking -c $cpus -g 0 --memory 8g -w $work_dir '$command'"
echo "nlprun --job-name summarize-$1 --priority high -a crfm_benchmarking -c $cpus -g 0 --memory 8g -w $work_dir 'benchmark-summarize --suite $1'"