////////////////////////////////////////////////////////////
// Helper functions for getting URLs of JSON files
function baseUrlWithDirectories(version, using_release) {
  parent_directory = (using_release ? 'releases' : 'runs');
  return `${BENCHMARK_OUTPUT_BASE_URL}/${parent_directory}/${version}`
}

function summaryJsonUrl(version, using_release) {
  return `${baseUrlWithDirectories(version, using_release)}/summary.json`;
}

function runsToRunSuitesJsonUrl(version, using_release) {
  return `${baseUrlWithDirectories(version, using_release)}/runs_to_run_suites.json`;
}

function runSpecsJsonUrl(version, using_release) {
  return `${baseUrlWithDirectories(version, using_release)}/run_specs.json`;
}

function groupsMetadataJsonUrl(version, using_release) {
  return `${baseUrlWithDirectories(version, using_release)}/groups_metadata.json`;
}

function groupsJsonUrl(version, using_release) {
  return `${baseUrlWithDirectories(version, using_release)}/groups.json`;
}

function groupJsonUrl(version, using_release, groupName) {
  return `${baseUrlWithDirectories(version, using_release)}/groups/${groupName}.json`;
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
