export { startTests } from '@vitest/runner';
import { setSafeTimers } from '@vitest/utils';
import { a as resetRunOnceCounter } from './vendor-run-once.1fa85ba7.js';
export { g as getCoverageProvider, s as startCoverageInsideWorker, a as stopCoverageInsideWorker, t as takeCoverageInsideWorker } from './vendor-coverage.2e41927a.js';
import './vendor-index.23ac4e13.js';
import 'pathe';
import 'std-env';
import '@vitest/runner/utils';
import './vendor-global.6795f91f.js';

let globalSetup = false;
async function setupCommonEnv(config) {
  resetRunOnceCounter();
  setupDefines(config.defines);
  if (globalSetup)
    return;
  globalSetup = true;
  setSafeTimers();
  if (config.globals)
    (await import('./chunk-integrations-globals.0093e2ed.js')).registerApiGlobally();
}
function setupDefines(defines) {
  for (const key in defines)
    globalThis[key] = defines[key];
}

export { setupCommonEnv };
