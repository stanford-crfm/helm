import { pathToFileURL } from 'node:url';
import { ModuleCacheMap, ViteNodeRunner } from 'vite-node/client';
import { isNodeBuiltin, isInternalRequest, isPrimitive } from 'vite-node/utils';
import { isAbsolute, dirname, join, basename, extname, resolve, normalize, relative } from 'pathe';
import { processError } from '@vitest/utils/error';
import { g as getWorkerState, a as getCurrentEnvironment } from './vendor-global.6795f91f.js';
import { d as distDir } from './vendor-paths.84fc7a99.js';
import { existsSync, readdirSync } from 'node:fs';
import { getColors, getType } from '@vitest/utils';
import { g as getAllMockableProperties } from './vendor-index.cc463d9e.js';
import { spyOn } from '@vitest/spy';
import { a as rpc } from './vendor-rpc.ad5b08c7.js';

const filterPublicKeys = ["__esModule", Symbol.asyncIterator, Symbol.hasInstance, Symbol.isConcatSpreadable, Symbol.iterator, Symbol.match, Symbol.matchAll, Symbol.replace, Symbol.search, Symbol.split, Symbol.species, Symbol.toPrimitive, Symbol.toStringTag, Symbol.unscopables];
class RefTracker {
  idMap = /* @__PURE__ */ new Map();
  mockedValueMap = /* @__PURE__ */ new Map();
  getId(value) {
    return this.idMap.get(value);
  }
  getMockedValue(id) {
    return this.mockedValueMap.get(id);
  }
  track(originalValue, mockedValue) {
    const newId = this.idMap.size;
    this.idMap.set(originalValue, newId);
    this.mockedValueMap.set(newId, mockedValue);
    return newId;
  }
}
function isSpecialProp(prop, parentType) {
  return parentType.includes("Function") && typeof prop === "string" && ["arguments", "callee", "caller", "length", "name"].includes(prop);
}
class VitestMocker {
  constructor(executor) {
    this.executor = executor;
  }
  static pendingIds = [];
  resolveCache = /* @__PURE__ */ new Map();
  get root() {
    return this.executor.options.root;
  }
  get mockMap() {
    return this.executor.options.mockMap;
  }
  get moduleCache() {
    return this.executor.moduleCache;
  }
  get moduleDirectories() {
    return this.executor.options.moduleDirectories || [];
  }
  deleteCachedItem(id) {
    const mockId = this.getMockPath(id);
    if (this.moduleCache.has(mockId))
      this.moduleCache.delete(mockId);
  }
  isAModuleDirectory(path) {
    return this.moduleDirectories.some((dir) => path.includes(dir));
  }
  getSuiteFilepath() {
    return getWorkerState().filepath || "global";
  }
  getMocks() {
    const suite = this.getSuiteFilepath();
    const suiteMocks = this.mockMap.get(suite);
    const globalMocks = this.mockMap.get("global");
    return {
      ...globalMocks,
      ...suiteMocks
    };
  }
  async resolvePath(rawId, importer) {
    let id;
    let fsPath;
    try {
      [id, fsPath] = await this.executor.originalResolveUrl(rawId, importer);
    } catch (error) {
      if (error.code === "ERR_MODULE_NOT_FOUND") {
        const { id: unresolvedId } = error[Symbol.for("vitest.error.not_found.data")];
        id = unresolvedId;
        fsPath = unresolvedId;
      } else {
        throw error;
      }
    }
    const external = !isAbsolute(fsPath) || this.isAModuleDirectory(fsPath) ? rawId : null;
    return {
      id,
      fsPath,
      external
    };
  }
  async resolveMocks() {
    if (!VitestMocker.pendingIds.length)
      return;
    await Promise.all(VitestMocker.pendingIds.map(async (mock) => {
      const { fsPath, external } = await this.resolvePath(mock.id, mock.importer);
      if (mock.type === "unmock")
        this.unmockPath(fsPath);
      if (mock.type === "mock")
        this.mockPath(mock.id, fsPath, external, mock.factory);
    }));
    VitestMocker.pendingIds = [];
  }
  async callFunctionMock(dep, mock) {
    var _a, _b;
    const cached = (_a = this.moduleCache.get(dep)) == null ? void 0 : _a.exports;
    if (cached)
      return cached;
    let exports;
    try {
      exports = await mock();
    } catch (err) {
      const vitestError = new Error(
        '[vitest] There was an error when mocking a module. If you are using "vi.mock" factory, make sure there are no top level variables inside, since this call is hoisted to top of the file. Read more: https://vitest.dev/api/vi.html#vi-mock'
      );
      vitestError.cause = err;
      throw vitestError;
    }
    const filepath = dep.slice(5);
    const mockpath = ((_b = this.resolveCache.get(this.getSuiteFilepath())) == null ? void 0 : _b[filepath]) || filepath;
    if (exports === null || typeof exports !== "object")
      throw new Error(`[vitest] vi.mock("${mockpath}", factory?: () => unknown) is not returning an object. Did you mean to return an object with a "default" key?`);
    const moduleExports = new Proxy(exports, {
      get(target, prop) {
        const val = target[prop];
        if (prop === "then") {
          if (target instanceof Promise)
            return target.then.bind(target);
        } else if (!(prop in target)) {
          if (filterPublicKeys.includes(prop))
            return void 0;
          const c = getColors();
          throw new Error(
            `[vitest] No "${String(prop)}" export is defined on the "${mockpath}" mock. Did you forget to return it from "vi.mock"?
If you need to partially mock a module, you can use "vi.importActual" inside:

${c.green(`vi.mock("${mockpath}", async () => {
  const actual = await vi.importActual("${mockpath}")
  return {
    ...actual,
    // your mocked methods
  },
})`)}
`
          );
        }
        return val;
      }
    });
    this.moduleCache.set(dep, { exports: moduleExports });
    return moduleExports;
  }
  getMockPath(dep) {
    return `mock:${dep}`;
  }
  getDependencyMock(id) {
    return this.getMocks()[id];
  }
  normalizePath(path) {
    return this.moduleCache.normalizePath(path);
  }
  resolveMockPath(mockPath, external) {
    const path = external || mockPath;
    if (external || isNodeBuiltin(mockPath) || !existsSync(mockPath)) {
      const mockDirname = dirname(path);
      const mockFolder = join(this.root, "__mocks__", mockDirname);
      if (!existsSync(mockFolder))
        return null;
      const files = readdirSync(mockFolder);
      const baseOriginal = basename(path);
      for (const file of files) {
        const baseFile = basename(file, extname(file));
        if (baseFile === baseOriginal)
          return resolve(mockFolder, file);
      }
      return null;
    }
    const dir = dirname(path);
    const baseId = basename(path);
    const fullPath = resolve(dir, "__mocks__", baseId);
    return existsSync(fullPath) ? fullPath : null;
  }
  mockObject(object, mockExports = {}) {
    const finalizers = new Array();
    const refs = new RefTracker();
    const define = (container, key, value) => {
      try {
        container[key] = value;
        return true;
      } catch {
        return false;
      }
    };
    const mockPropertiesOf = (container, newContainer) => {
      const containerType = getType(container);
      const isModule = containerType === "Module" || !!container.__esModule;
      for (const { key: property, descriptor } of getAllMockableProperties(container, isModule)) {
        if (!isModule && descriptor.get) {
          try {
            Object.defineProperty(newContainer, property, descriptor);
          } catch (error) {
          }
          continue;
        }
        if (isSpecialProp(property, containerType))
          continue;
        const value = container[property];
        const refId = refs.getId(value);
        if (refId !== void 0) {
          finalizers.push(() => define(newContainer, property, refs.getMockedValue(refId)));
          continue;
        }
        const type = getType(value);
        if (Array.isArray(value)) {
          define(newContainer, property, []);
          continue;
        }
        const isFunction = type.includes("Function") && typeof value === "function";
        if ((!isFunction || value.__isMockFunction) && type !== "Object" && type !== "Module") {
          define(newContainer, property, value);
          continue;
        }
        if (!define(newContainer, property, isFunction ? value : {}))
          continue;
        if (isFunction) {
          const mock = spyOn(newContainer, property).mockImplementation(() => void 0);
          mock.mockRestore = () => {
            mock.mockReset();
            mock.mockImplementation(() => void 0);
            return mock;
          };
          Object.defineProperty(newContainer[property], "length", { value: 0 });
        }
        refs.track(value, newContainer[property]);
        mockPropertiesOf(value, newContainer[property]);
      }
    };
    const mockedObject = mockExports;
    mockPropertiesOf(object, mockedObject);
    for (const finalizer of finalizers)
      finalizer();
    return mockedObject;
  }
  unmockPath(path) {
    const suitefile = this.getSuiteFilepath();
    const id = this.normalizePath(path);
    const mock = this.mockMap.get(suitefile);
    if (mock && id in mock)
      delete mock[id];
    this.deleteCachedItem(id);
  }
  mockPath(originalId, path, external, factory) {
    const suitefile = this.getSuiteFilepath();
    const id = this.normalizePath(path);
    const mocks = this.mockMap.get(suitefile) || {};
    const resolves = this.resolveCache.get(suitefile) || {};
    mocks[id] = factory || this.resolveMockPath(path, external);
    resolves[id] = originalId;
    this.mockMap.set(suitefile, mocks);
    this.resolveCache.set(suitefile, resolves);
    this.deleteCachedItem(id);
  }
  async importActual(rawId, importee) {
    const { id, fsPath } = await this.resolvePath(rawId, importee);
    const result = await this.executor.cachedRequest(id, fsPath, [importee]);
    return result;
  }
  async importMock(rawId, importee) {
    const { id, fsPath, external } = await this.resolvePath(rawId, importee);
    const normalizedId = this.normalizePath(fsPath);
    let mock = this.getDependencyMock(normalizedId);
    if (mock === void 0)
      mock = this.resolveMockPath(fsPath, external);
    if (mock === null) {
      const mod = await this.executor.cachedRequest(id, fsPath, [importee]);
      return this.mockObject(mod);
    }
    if (typeof mock === "function")
      return this.callFunctionMock(fsPath, mock);
    return this.executor.dependencyRequest(mock, mock, [importee]);
  }
  async requestWithMock(url, callstack) {
    const id = this.normalizePath(url);
    const mock = this.getDependencyMock(id);
    const mockPath = this.getMockPath(id);
    if (mock === null) {
      const cache = this.moduleCache.get(mockPath);
      if (cache.exports)
        return cache.exports;
      const exports = {};
      this.moduleCache.set(mockPath, { exports });
      const mod = await this.executor.directRequest(url, url, callstack);
      this.mockObject(mod, exports);
      return exports;
    }
    if (typeof mock === "function" && !callstack.includes(mockPath) && !callstack.includes(url)) {
      callstack.push(mockPath);
      const result = await this.callFunctionMock(mockPath, mock);
      const indexMock = callstack.indexOf(mockPath);
      callstack.splice(indexMock, 1);
      return result;
    }
    if (typeof mock === "string" && !callstack.includes(mock))
      return mock;
  }
  queueMock(id, importer, factory) {
    VitestMocker.pendingIds.push({ type: "mock", id, importer, factory });
  }
  queueUnmock(id, importer) {
    VitestMocker.pendingIds.push({ type: "unmock", id, importer });
  }
}

async function createVitestExecutor(options) {
  const runner = new VitestExecutor(options);
  await runner.executeId("/@vite/env");
  return runner;
}
let _viteNode;
const moduleCache = new ModuleCacheMap();
const mockMap = /* @__PURE__ */ new Map();
async function startViteNode(ctx) {
  if (_viteNode)
    return _viteNode;
  const { config } = ctx;
  const processExit = process.exit;
  process.exit = (code = process.exitCode || 0) => {
    const error = new Error(`process.exit called with "${code}"`);
    rpc().onWorkerExit(error, code);
    return processExit(code);
  };
  function catchError(err, type) {
    var _a;
    const worker = getWorkerState();
    const error = processError(err);
    if (!isPrimitive(error)) {
      error.VITEST_TEST_NAME = (_a = worker.current) == null ? void 0 : _a.name;
      if (worker.filepath)
        error.VITEST_TEST_PATH = relative(config.root, worker.filepath);
      error.VITEST_AFTER_ENV_TEARDOWN = worker.environmentTeardownRun;
    }
    rpc().onUnhandledError(error, type);
  }
  process.on("uncaughtException", (e) => catchError(e, "Uncaught Exception"));
  process.on("unhandledRejection", (e) => catchError(e, "Unhandled Rejection"));
  const executor = await createVitestExecutor({
    fetchModule(id) {
      return rpc().fetch(id, ctx.environment.name);
    },
    resolveId(id, importer) {
      return rpc().resolveId(id, importer, ctx.environment.name);
    },
    moduleCache,
    mockMap,
    interopDefault: config.deps.interopDefault,
    moduleDirectories: config.deps.moduleDirectories,
    root: config.root,
    base: config.base
  });
  const { run } = await import(pathToFileURL(resolve(distDir, "entry.js")).href);
  _viteNode = { run, executor };
  return _viteNode;
}
class VitestExecutor extends ViteNodeRunner {
  constructor(options) {
    super(options);
    this.options = options;
    this.mocker = new VitestMocker(this);
    Object.defineProperty(globalThis, "__vitest_mocker__", {
      value: this.mocker,
      writable: true,
      configurable: true
    });
  }
  mocker;
  shouldResolveId(id, _importee) {
    if (isInternalRequest(id) || id.startsWith("data:"))
      return false;
    const environment = getCurrentEnvironment();
    return environment === "node" ? !isNodeBuiltin(id) : !id.startsWith("node:");
  }
  async originalResolveUrl(id, importer) {
    return super.resolveUrl(id, importer);
  }
  async resolveUrl(id, importer) {
    if (VitestMocker.pendingIds.length)
      await this.mocker.resolveMocks();
    if (importer && importer.startsWith("mock:"))
      importer = importer.slice(5);
    try {
      return await super.resolveUrl(id, importer);
    } catch (error) {
      if (error.code === "ERR_MODULE_NOT_FOUND") {
        const { id: id2 } = error[Symbol.for("vitest.error.not_found.data")];
        const path = this.mocker.normalizePath(id2);
        const mock = this.mocker.getDependencyMock(path);
        if (mock !== void 0)
          return [id2, id2];
      }
      throw error;
    }
  }
  async dependencyRequest(id, fsPath, callstack) {
    const mocked = await this.mocker.requestWithMock(fsPath, callstack);
    if (typeof mocked === "string")
      return super.dependencyRequest(mocked, mocked, callstack);
    if (mocked && typeof mocked === "object")
      return mocked;
    return super.dependencyRequest(id, fsPath, callstack);
  }
  prepareContext(context) {
    const workerState = getWorkerState();
    if (workerState.filepath && normalize(workerState.filepath) === normalize(context.__filename)) {
      Object.defineProperty(context.__vite_ssr_import_meta__, "vitest", { get: () => globalThis.__vitest_index__ });
    }
    return context;
  }
}

export { VitestExecutor as V, mockMap as a, moduleCache as m, startViteNode as s };
