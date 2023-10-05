import { V as VitestRunMode, U as UserConfig, o as Vitest, p as PendingSuiteMock, M as MockFactory, q as MockMap, r as TestSequencer, W as WorkspaceSpec } from './types-198fd1d9.js';
export { u as TestSequencerConstructor, s as VitestWorkspace, t as startVitest } from './types-198fd1d9.js';
import { UserConfig as UserConfig$1, Plugin } from 'vite';
import { ViteNodeRunner } from 'vite-node/client';
import { ViteNodeRunnerOptions } from 'vite-node';
import '@vitest/snapshot';
import '@vitest/expect';
import '@vitest/runner';
import '@vitest/runner/utils';
import '@vitest/utils';
import 'tinybench';
import '@vitest/snapshot/manager';
import 'vite-node/server';
import 'node:worker_threads';
import 'node:fs';
import 'chai';

declare function createVitest(mode: VitestRunMode, options: UserConfig, viteOverrides?: UserConfig$1): Promise<Vitest>;

declare function VitestPlugin(options?: UserConfig, ctx?: Vitest): Promise<Plugin[]>;

declare function registerConsoleShortcuts(ctx: Vitest): () => void;

type Key = string | symbol;
declare class VitestMocker {
    executor: VitestExecutor;
    static pendingIds: PendingSuiteMock[];
    private resolveCache;
    constructor(executor: VitestExecutor);
    private get root();
    private get mockMap();
    private get moduleCache();
    private get moduleDirectories();
    private deleteCachedItem;
    private isAModuleDirectory;
    getSuiteFilepath(): string;
    getMocks(): {
        [x: string]: string | MockFactory | null;
    };
    private resolvePath;
    resolveMocks(): Promise<void>;
    private callFunctionMock;
    getMockPath(dep: string): string;
    getDependencyMock(id: string): string | MockFactory | null;
    normalizePath(path: string): string;
    resolveMockPath(mockPath: string, external: string | null): string | null;
    mockObject(object: Record<Key, any>, mockExports?: Record<Key, any>): Record<Key, any>;
    unmockPath(path: string): void;
    mockPath(originalId: string, path: string, external: string | null, factory?: MockFactory): void;
    importActual<T>(rawId: string, importee: string): Promise<T>;
    importMock(rawId: string, importee: string): Promise<any>;
    requestWithMock(url: string, callstack: string[]): Promise<any>;
    queueMock(id: string, importer: string, factory?: MockFactory): void;
    queueUnmock(id: string, importer: string): void;
}

interface ExecuteOptions extends ViteNodeRunnerOptions {
    mockMap: MockMap;
    moduleDirectories?: string[];
}
declare class VitestExecutor extends ViteNodeRunner {
    options: ExecuteOptions;
    mocker: VitestMocker;
    constructor(options: ExecuteOptions);
    shouldResolveId(id: string, _importee?: string | undefined): boolean;
    originalResolveUrl(id: string, importer?: string): Promise<[url: string, fsPath: string]>;
    resolveUrl(id: string, importer?: string): Promise<[string, string]>;
    dependencyRequest(id: string, fsPath: string, callstack: string[]): Promise<any>;
    prepareContext(context: Record<string, any>): Record<string, any>;
}

declare class BaseSequencer implements TestSequencer {
    protected ctx: Vitest;
    constructor(ctx: Vitest);
    shard(files: WorkspaceSpec[]): Promise<WorkspaceSpec[]>;
    sort(files: WorkspaceSpec[]): Promise<WorkspaceSpec[]>;
}

export { BaseSequencer, ExecuteOptions, TestSequencer, Vitest, VitestExecutor, VitestPlugin, WorkspaceSpec, createVitest, registerConsoleShortcuts };
