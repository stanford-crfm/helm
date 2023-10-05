import { performance } from 'node:perf_hooks';
import v8 from 'node:v8';
import { c as createBirpc } from './vendor-index.cc463d9e.js';
import { parseRegexp } from '@vitest/utils';
import { s as startViteNode, m as moduleCache, a as mockMap } from './vendor-execute.3576af13.js';
import { r as rpcDone } from './vendor-rpc.ad5b08c7.js';
import { s as setupInspect } from './vendor-inspector.47fc8cbb.js';
import 'node:url';
import 'vite-node/client';
import 'vite-node/utils';
import 'pathe';
import '@vitest/utils/error';
import './vendor-global.6795f91f.js';
import './vendor-paths.84fc7a99.js';
import 'node:fs';
import '@vitest/spy';
import 'node:module';

function init(ctx) {
  const { config } = ctx;
  process.env.VITEST_WORKER_ID = "1";
  process.env.VITEST_POOL_ID = "1";
  let setCancel = (_reason) => {
  };
  const onCancel = new Promise((resolve) => {
    setCancel = resolve;
  });
  globalThis.__vitest_environment__ = config.environment;
  globalThis.__vitest_worker__ = {
    ctx,
    moduleCache,
    config,
    mockMap,
    onCancel,
    durations: {
      environment: 0,
      prepare: performance.now()
    },
    rpc: createBirpc(
      {
        onCancel: setCancel
      },
      {
        eventNames: ["onUserConsoleLog", "onFinished", "onCollected", "onWorkerExit", "onCancel"],
        serialize: v8.serialize,
        deserialize: (v) => v8.deserialize(Buffer.from(v)),
        post(v) {
          var _a;
          (_a = process.send) == null ? void 0 : _a.call(process, v);
        },
        on(fn) {
          process.on("message", fn);
        }
      }
    )
  };
  if (ctx.invalidates) {
    ctx.invalidates.forEach((fsPath) => {
      moduleCache.delete(fsPath);
      moduleCache.delete(`mock:${fsPath}`);
    });
  }
  ctx.files.forEach((i) => moduleCache.delete(i));
}
function parsePossibleRegexp(str) {
  const prefix = "$$vitest:";
  if (typeof str === "string" && str.startsWith(prefix))
    return parseRegexp(str.slice(prefix.length));
  return str;
}
function unwrapConfig(config) {
  if (config.testNamePattern)
    config.testNamePattern = parsePossibleRegexp(config.testNamePattern);
  return config;
}
async function run(ctx) {
  const inspectorCleanup = setupInspect(ctx.config);
  try {
    init(ctx);
    const { run: run2, executor } = await startViteNode(ctx);
    await run2(ctx.files, ctx.config, ctx.environment, executor);
    await rpcDone();
  } finally {
    inspectorCleanup();
  }
}
const procesExit = process.exit;
process.on("message", async (message) => {
  if (typeof message === "object" && message.command === "start") {
    try {
      message.config = unwrapConfig(message.config);
      await run(message);
    } finally {
      procesExit();
    }
  }
});

export { run };
