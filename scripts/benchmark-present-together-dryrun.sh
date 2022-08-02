: '
Does a dryrun for all the together models in parallel.
The dry run results will be outputted to benchmark_output/runs/together.

Usage:

  bash scripts/benchmark-present-together-dryrun.sh <Any additional CLI arguments for benchmark-present>

  e.g.,
  bash scripts/benchmark-present-together-dryrun.sh --max-eval-instances 1000 --priority 2 --local

To kill a running process:

  ps -A | grep benchmark-present
  kill <pid>
'

function execute {
   # Prints and executes command
   echo $1
   eval "time $1"
}

models=(
  "together/bloom"
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
    logfile="${model//\//-}"  # Replace slashes
    logfile="${logfile// /_}"   # Replace spaces

    # Override with passed-in CLI arguments
    # By default, the command will run the RunSpecs listed in src/benchmark/presentation/run_specs.conf
    # and output results to `benchmark_output/runs/together`.
    execute "benchmark-present --suite together --dry-run --models-to-run $model $* &> dryrun_$logfile.log &"
done
