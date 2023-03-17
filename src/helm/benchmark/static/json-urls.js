////////////////////////////////////////////////////////////
// Helper functions for getting URLs of JSON files

function summaryJsonUrl(suite) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/summary.json`;
}

function runSpecsJsonUrl(suite) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/run_specs.json`;
}

function groupsMetadataJsonUrl(suite) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/groups_metadata.json`;
}

function groupsJsonUrl(suite) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/groups.json`;
}

function groupJsonUrl(suite, groupName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/groups/${groupName}.json`;
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
