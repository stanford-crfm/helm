import { N as Nullable, A as Arrayable } from './types-516036fa.js';
import 'vite/types/hot';
import './types.d-7442d07f.js';

declare const isWindows: boolean;
declare function slash(str: string): string;
declare const VALID_ID_PREFIX = "/@id/";
declare function normalizeRequestId(id: string, base?: string): string;
declare const queryRE: RegExp;
declare const hashRE: RegExp;
declare function cleanUrl(url: string): string;
declare function isInternalRequest(id: string): boolean;
declare function normalizeModuleId(id: string): string;
declare function isPrimitive(v: any): boolean;
declare function toFilePath(id: string, root: string): {
    path: string;
    exists: boolean;
};
declare function isNodeBuiltin(id: string): boolean;
/**
 * Convert `Arrayable<T>` to `Array<T>`
 *
 * @category Array
 */
declare function toArray<T>(array?: Nullable<Arrayable<T>>): Array<T>;

export { VALID_ID_PREFIX, cleanUrl, hashRE, isInternalRequest, isNodeBuiltin, isPrimitive, isWindows, normalizeModuleId, normalizeRequestId, queryRE, slash, toArray, toFilePath };
