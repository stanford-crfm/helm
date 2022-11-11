#!/bin/bash

if [ $# != 1 ]; then
  echo Copy everything into a www directory.
  echo "Usage: `basename $0` <suite>"
  exit 1
fi

suite=$1; shift

echo Copying suite $suite into www...

# Copy code (note: follow symlinks)
mkdir -p www || exit 1
rsync -pLrvz --exclude=benchmark_output src/benchmark/static/* www || exit 1

# Copy data
mkdir -p www/benchmark_output/runs || exit 1
cp -a benchmark_output/runs/$suite www/benchmark_output/runs || exit 1
ln -sf $suite www/benchmark_output/runs/latest || exit 1

# Set permissions
chmod -R og=u-w www || exit 1

# Push to server (https://nlp.stanford.edu/helm)
echo Pushing to server...
rsync -arvz www/* /nlp/scr2/htdocs-helm/$suite
