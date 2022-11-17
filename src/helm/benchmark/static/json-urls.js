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

function perInstanceStatsJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/per_instance_stats_slim.json`;
}

function runSpecJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/run_spec.json`
}

function scenarioJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/scenario.json`;
}

function scenarioStateJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/scenario_state_slim.json`;
}

function statsJsonUrl(suite, runSpecName) {
  return `${BENCHMARK_OUTPUT_BASE_URL}/runs/${suite}/${runSpecName}/stats.json`;
}
