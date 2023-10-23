////////////////////////////////////////////////////////////
// Helper functions for getting URLs of JSON files
function versionBaseUrl() {
  if (window.RELEASE) {
    return `${BENCHMARK_OUTPUT_BASE_URL}/releases/${window.RELEASE}`;
  } else {
    return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${window.SUITE}`;
  }
}

function summaryJsonUrl() {
  return `${versionBaseUrl()}/summary.json`;
}

function runsToRunSuitesJsonUrl() {
  return `${versionBaseUrl()}/runs_to_run_suites.json`;
}

function runSpecsJsonUrl() {
  return `${versionBaseUrl()}/run_specs.json`;
}

function groupsMetadataJsonUrl() {
  return `${versionBaseUrl(version, using_release)}/groups_metadata.json`;
}

function groupsJsonUrl() {
  return `${versionBaseUrl()}/groups.json`;
}

function groupJsonUrl(groupName) {
  return `${versionBaseUrl()}/groups/${groupName}.json`;
}

function runSpecJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/run_spec.json`;
}

function scenarioJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/scenario.json`;
}

function scenarioStateJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/scenario_state.json`;
}

function statsJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/stats.json`;
}

function instancesJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/instances.json`;
}

function predictionsJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/display_predictions.json`;
}

function requestsJsonUrl(runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/display_requests.json`;
}

function plotUrl(plotName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/plots/${plotName}.png`;
}
