function getWorkerState() {
  return globalThis.__vitest_worker__;
}
function getCurrentEnvironment() {
  return globalThis.__vitest_environment__;
}

export { getCurrentEnvironment as a, getWorkerState as g };
