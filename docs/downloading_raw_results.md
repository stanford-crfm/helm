# Downloading Raw Results

All of HELM's raw result data is stored in Google Cloud Storage (GCS) in the public `crfm-helm-public` bucket. If you wish to download the raw result data, you can use Google Cloud's tools to do so. The following walks through how to use the `gsutil` tool ([documentation](https://cloud.google.com/storage/docs/gsutil)) to download the data.

## Setup

1. Follow [Google's installation instructions](https://cloud.google.com/storage/docs/gsutil_install) to install `gsutil`. When prompted "Would you like to log in (Y/n)?", respond with no, because the HELM GCS bucket is public and does not require credentials.
2. Create a local directory to store the data:
```sh
export LOCAL_BENCHMARK_OUTPUT_PATH=./benchmark_output
mkdir $LOCAL_BENCHMARK_OUTPUT_PATH
```
3. Set the Google Cloud Storage path to the appropriate path:
```sh
export GCS_BENCHMARK_OUTPUT_PATH=gs://crfm-helm-public/lite/benchmark_output
```

## Paths

Locations of the `benchmark_output` folders for each project:

- Lite: `gs://crfm-helm-public/lite/benchmark_output`
- Classic: `gs://crfm-helm-public/benchmark_output` (see warning above)
- HEIM: `gs://crfm-helm-public/heim/benchmark_output`
- Instruct: `gs://crfm-helm-public/instruct/benchmark_output`

## Download a whole project

Warning: Downloading a whole HELM project requires a very large amounts of disk space - a few hundred GB for most projects, and more than 1 TB for Classic. Ensure that you have enough local disk space before downloading these projects.

1. (Optional) Use the `gsutil du` ([documentation](https://cloud.google.com/storage/docs/gsutil/commands/du)) command to compute the size of the download and ensure you have enough space on your local disk:
```sh
gsutil du -sh $GCS_BENCHMARK_OUTPUT_PATH
```
2. Run `gsutil rsync` ([documentation](https://cloud.google.com/storage/docs/gsutil/commands/rsync)) to download the data to the folder created in the previous step:
```sh
gsutil -m rsync -r $GCS_BENCHMARK_OUTPUT_PATH $LOCAL_BENCHMARK_OUTPUT_PATH
```

## Download a specific version

You can also download a specific version to save local disk space. The instructions differ depending on which version you are downloading. Check if your version is in the following list:

- Lite: all versions
- Classic: v0.3.0 or later

If you are downloading one of the above versions, then you must follow the instructions in _both_ **Download a specific releases** and **Download a specific suite**. Otherwise, you should _skip_ **Download a specific releases** and _only_ follow the instructions in **Download a specific suite**.

## Download a specific release

1. Set the release version:
```sh
export RELEASE_VERSION=v1.0.0
```
2. Create a local directory to store the data:
```sh
mkdir $LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION
```
3. Run `gsutil rsync` ([documentation](https://cloud.google.com/storage/docs/gsutil/commands/rsync)) to download the data to the folder created in the previous step:
```sh
gsutil -m rsync -r $GCS_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION $LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION
```
4. Inspect the file contents of `$LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION/summary.json`. For _each_ suite listed in the `suites` array field, repeat the steps in **Download a specific suite** for that suite.

## Download a specific suite

1. Set the suite version:
```sh
export SUITE_VERSION=v1.0.0
```
2. Create a local directory to store the data:
```sh
mkdir $LOCAL_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION
```
3. Run `gsutil rsync` ([documentation](https://cloud.google.com/storage/docs/gsutil/commands/rsync)) to download the data to the folder created in the previous step:
```sh
gsutil -m rsync -r $GCS_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION $LOCAL_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION
```

## GCS browser

If you wish to explore the raw data files in the web browser without downloading it, you can use the [GCS browser](https://console.cloud.google.com/storage/browser/crfm-helm-public). Note that this requires logging into any Google account and agreeing to the Google Cloud Platform Terms of Service.
