////////////////////////////////////////////////////////////
// Helper functions for getting URLs of JSON files

function runManifestJsonUrl(release) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${release}/run_manifest.json`;
}

function summaryJsonUrl(release) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${release}/summary.json`;
}

function runSpecsJsonUrl(release) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${release}/run_specs.json`;
}

function groupsMetadataJsonUrl(release) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${release}/groups_metadata.json`;
}

function groupsJsonUrl(release) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${release}/groups.json`;
}

function groupJsonUrl(release, groupName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${release}/groups/${groupName}.json`;
}

function runSpecJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/run_spec.json`
}

function scenarioJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/scenario.json`;
}

function scenarioStateJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/scenario_state.json`;
}

function statsJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/stats.json`;
}

function instancesJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/instances.json`;
}

function predictionsJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/display_predictions.json`;
}

function requestsJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/display_requests.json`;
}

function plotUrl(suite, plotName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/plots/${plotName}.png`;
}
