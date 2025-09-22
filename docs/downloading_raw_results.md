# Downloading Raw Results

All of HELM's raw result data is stored in Google Cloud Storage (GCS) in the public `crfm-helm-public` bucket. If you wish to download the raw result data, you can use the Google Cloud Platform (GCP) tools to do so. The following walks through how to use the `gcloud storage` command line tool ([documentation](https://cloud.google.com/sdk/gcloud/reference/storage)) to download the data.

## Setup

1. Follow [Google's installation instructions](https://cloud.google.com/sdk/docs/install) to install `gcloud`. If the installer prompts you to log in, you may skip this step because the HELM GCS bucket allows public unauthenticated access.
2. Create a local directory to store the data:
```sh
export LOCAL_BENCHMARK_OUTPUT_PATH=./benchmark_output
mkdir $LOCAL_BENCHMARK_OUTPUT_PATH
```
3. Set the GCS path to the appropriate path:
```sh
export GCS_BENCHMARK_OUTPUT_PATH=gs://crfm-helm-public/lite/benchmark_output
```

## Paths

Locations of the `benchmark_output` folders for each project:

- Capabilities: `gs://crfm-helm-public/capabilities/benchmark_output`
- Safety: `gs://crfm-helm-public/safety/benchmark_output`
- AIR-Bench: `gs://crfm-helm-public/air-bench/benchmark_output`
- Lite: `gs://crfm-helm-public/lite/benchmark_output`
- MMLU: `gs://crfm-helm-public/mmlu/benchmark_output`
- Classic: `gs://crfm-helm-public/benchmark_output` (see warning above)
- HEIM: `gs://crfm-helm-public/heim/benchmark_output`
- Instruct: `gs://crfm-helm-public/instruct/benchmark_output`
- MedHELM: `gs://crfm-helm-public/medhelm/benchmark_output`
- ToRR: `gs://crfm-helm-public/torr/benchmark_output`
- VHELM: `gs://crfm-helm-public/vhelm/benchmark_output`
- AHELM: `gs://crfm-helm-public/audio/benchmark_output`
- Image2Struct: `gs://crfm-helm-public/image2struct/benchmark_output`

## Download a whole project

Warning: Downloading a whole HELM project requires a very large amounts of disk space - a few hundred GB for most projects, and more than 1 TB for Classic. Ensure that you have enough local disk space before downloading these projects.

1. (Optional) Use the `gcloud storage du` ([documentation](https://cloud.google.com/sdk/gcloud/reference/storage/du)) command to compute the size of the download and ensure you have enough space on your local disk:
```sh
gcloud storage du -sh $GCS_BENCHMARK_OUTPUT_PATH
```
2. Run `gcloud storage rsync` ([documentation](https://cloud.google.com/sdk/gcloud/reference/storage/rsync)) to download the data to the folder created in the previous step:
```sh
gcloud storage rsync -r $GCS_BENCHMARK_OUTPUT_PATH $LOCAL_BENCHMARK_OUTPUT_PATH
```

## Download a specific version

You can also download a specific version to save local disk space. The instructions differ depending on which version you are downloading. Check if your version is in the following list:

- Classic: before v0.3.0
- VHELM
- AHELM
- Image2Struct

If you are downloading one of the above versions, then you should _skip_ **Download a specific releases** and _only_ follow the instructions in **Download a specific suite**.

Otherwise, you should follow the instructions in _both_ **Download a specific releases** and **Download a specific suite**.

## Download a specific release

1. Set the release version:
```sh
export RELEASE_VERSION=v1.0.0
```
2. Create a local directory to store the data:
```sh
mkdir $LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION
```
3. Run `gcloud storage rsync` ([documentation](https://cloud.google.com/sdk/gcloud/reference/storage/du)) to download the data to the folder created in the previous step:
```sh
gcloud storage rsync -r $GCS_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION $LOCAL_BENCHMARK_OUTPUT_PATH/releases/$RELEASE_VERSION
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
3. Run `gcloud storage rsync` ([documentation](https://cloud.google.com/sdk/gcloud/reference/storage/du)) to download the data to the folder created in the previous step:
```sh
gcloud storage rsync -r $GCS_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION $LOCAL_BENCHMARK_OUTPUT_PATH/runs/$SUITE_VERSION
```

## Troubleshooting

If you are on an older version of `gcloud`, you may encounter the error messages `(gcloud) Invalid choice: 'du'.` or `(gcloud) Invalid choice: 'rsync'.`. If so, you should either upgrade your `gcloud` installation to the latest version, or you may use the deprecated `gsutil` CLI tool ([documentation](https://cloud.google.com/storage/docs/gsutil)) instead.

To use `gsutil`, install gsutil following [Google's instructions](https://cloud.google.com/storage/docs/gsutil_install), then use the above command with `gcloud storage du` replaced with `gsutil du` ([documentation](https://cloud.google.com/storage/docs/gsutil/commands/du)) and `gcloud storage rsync` replaced with `gsutil rsync` ([documentation](https://cloud.google.com/storage/docs/gsutil/commands/rsync)).

## GCS browser

If you wish to explore the raw data files in the web browser without downloading it, you can use the [GCS browser](https://console.cloud.google.com/storage/browser/crfm-helm-public). Note that this requires logging into any Google account and agreeing to the GCP Terms of Service.
