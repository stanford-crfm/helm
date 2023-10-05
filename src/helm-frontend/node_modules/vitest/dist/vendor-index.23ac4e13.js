import { relative } from 'pathe';
import 'std-env';
import '@vitest/runner/utils';
import '@vitest/utils';
import { g as getWorkerState } from './vendor-global.6795f91f.js';

var _a;
const isNode = typeof process < "u" && typeof process.stdout < "u" && !((_a = process.versions) == null ? void 0 : _a.deno) && !globalThis.window;

const isWindows = isNode && process.platform === "win32";
function getRunMode() {
  return getWorkerState().config.mode;
}
function isRunningInBenchmark() {
  return getRunMode() === "benchmark";
}
const relativePath = relative;
function resetModules(modules, resetMocks = false) {
  const skipPaths = [
    // Vitest
    /\/vitest\/dist\//,
    /\/vite-node\/dist\//,
    // yarn's .store folder
    /vitest-virtual-\w+\/dist/,
    // cnpm
    /@vitest\/dist/,
    // don't clear mocks
    ...!resetMocks ? [/^mock:/] : []
  ];
  modules.forEach((mod, path) => {
    if (skipPaths.some((re) => re.test(path)))
      return;
    modules.invalidateModule(mod);
  });
}
function removeUndefinedValues(obj) {
  for (const key in Object.keys(obj)) {
    if (obj[key] === void 0)
      delete obj[key];
  }
  return obj;
}

export { isNode as a, relativePath as b, removeUndefinedValues as c, isWindows as d, isRunningInBenchmark as i, resetModules as r };
