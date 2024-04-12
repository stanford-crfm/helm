# Run Entries Configuration Files

In the tutorial, we have been using `--run-entries` to specify run entries for `helm-run`. However, we can also put the run entries into a **run entries configuration file**, and then pass the file to `helm-run` using the `--conf-file` flag.

This has a number of advantages:

- This prevents the command line invocation of `helm-run` from getting too long when a large number of run entries are run.
- The run entries configuration file can be shared with other users and commited to Git.

For example, instead of running:

```bash
helm-run --run-specs mmlu:subject=anatomy,model=openai/gpt2 mmlu:subject=philosophy,model=openai/gpt2 --suite tutorial --max-eval-instances 10
```

You can instead create a `tutorial_run_entries.conf` file in your current working directory:

```conf
entries: [
  {description: "mmlu:subject=anatomy,model=openai/gpt2", priority: 1},
  {description: "mmlu:subject=philosophy,model=openai/gpt2", priority: 1},
]
```

You would then use this file with `helm-run` with this command:

```bash
helm-run --conf-file tutorial_run_entries.conf --suite tutorial --max-eval-instances 10
```

## Model run expander wildcards

It is very common to use run entries configuration file with a **model run expander wildcards** e.g. `model=text`. For instance, 

```conf
entries: [
  {description: "mmlu:subject=anatomy,model=text", priority: 1},
  {description: "mmlu:subject=philosophy,model=text", priority: 1},
]
```

You would then use this file with `helm-run` with this command:

```bash
helm-run --conf-file tutorial_run_entries.conf --suite tutorial --max-eval-instances 10 --models-to-run openai/gpt2
```

This has exactly the same behavior has the previous example. For more information on model run expander wildcards, refer to the run entry format documentation.

## Priorities 

You can use the `--priority` flag in conjunction with `--conf-file`. This filters out run entries with a higher priority value than the specified `--priority` value. For instance, with this run entries configuration file:

```conf
entries: [
  {description: "mmlu:subject=anatomy,model=openai/gpt2", priority: 1},
  {description: "mmlu:subject=philosophy,model=openai/gpt2", priority: 2},
]
```

If run with `--priority 1`, only the first run entry will be run, and the second will be filtered out. If run with `--priority 2`, both run entries will be run.
