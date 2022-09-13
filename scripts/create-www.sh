#!/bin/bash

if [ $# != 1 ]; then
  echo Copy everything into a www directory.
  echo "Usage: `basename $0` <suite>"
  exit 1
fi

suite=$1; shift

echo Copying suite $suite into www...

# Copy code (note the renaming)
mkdir -p www || exit 1
rsync -arvz --exclude=benchmark_output src/proxy/static/* www || exit 1
ln -sf benchmarking.html www/index.html || exit 1

# Copy data
mkdir -p www/benchmark_output/runs || exit 1
cp -a benchmark_output/runs/$suite www/benchmark_output/runs || exit 1
ln -sf $suite www/benchmark_output/runs/latest || exit 1

# Set permissions
chmod -R og=u-w www || exit 1

# Push to server
echo Pushing to server...
rsync -arvz www/* scdt:/u/apache/htdocs/pliang/benchmarking
