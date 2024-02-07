#!/bin/bash
# Usage: sh create-www.sh $SUITE

if [ $# != 1 ]; then
  echo Copy everything into the www directory.
  echo "Usage: `basename $0` <suite>"
  exit 1
fi

suite=$1; shift

echo Copying suite $suite into www...

# Copy code (note: follow symlinks)
mkdir -p www || exit 1
rsync -pLrvz --exclude=benchmark_output src/helm/benchmark/static/* www || exit 1

# Copy data
mkdir -p www/benchmark_output/runs || exit 1
rsync -arvz benchmark_output/runs/$suite www/benchmark_output/runs || exit 1
ln -sf $suite www/benchmark_output/runs/latest || exit 1

# Set permissions
chmod -R og=u-w www || exit 1

# Make it live on the NLP website at https://nlp.stanford.edu/helm/<suite>
rsync -arvz www/* /u/apache/htdocs/helm/$suite

# Make it live
echo To make the contents of www live, do a git push or run something like:
echo
echo python3 /nlp/scr2/nlp/crfm/benchmarking/helm-deployment/gcs/upload.py --compress --bucket crfm-helm-public --source /nlp/scr2/nlp/crfm/benchmarking/benchmarking/www/benchmark_output --destination benchmark_output --source_suite $suite --destination_suite $suite
echo "Done."
