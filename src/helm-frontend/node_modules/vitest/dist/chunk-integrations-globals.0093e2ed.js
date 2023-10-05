import { g as globalApis } from './vendor-constants.538d9b49.js';
import { i as index } from './vendor-index.2af39fbb.js';
import '@vitest/runner';
import './vendor-vi.dd6706cb.js';
import '@vitest/runner/utils';
import '@vitest/utils';
import './vendor-index.23ac4e13.js';
import 'pathe';
import 'std-env';
import './vendor-global.6795f91f.js';
import 'chai';
import './vendor-_commonjsHelpers.7d1333e8.js';
import '@vitest/expect';
import '@vitest/snapshot';
import '@vitest/utils/error';
import './vendor-tasks.f9d75aed.js';
import 'util';
import '@vitest/spy';
import './vendor-run-once.1fa85ba7.js';

function registerApiGlobally() {
  globalApis.forEach((api) => {
    globalThis[api] = index[api];
  });
}

export { registerApiGlobally };
