/**
 * @typedef {(error?: Error|null|undefined, ...output: Array<any>) => void} Callback
 * @typedef {(...input: Array<any>) => any} Middleware
 *
 * @typedef {(...input: Array<any>) => void} Run
 *   Call all middleware.
 * @typedef {(fn: Middleware) => Pipeline} Use
 *   Add `fn` (middleware) to the list.
 * @typedef {{run: Run, use: Use}} Pipeline
 *   Middleware.
 */
/**
 * Create new middleware.
 *
 * @returns {Pipeline}
 */
export function trough(): Pipeline
/**
 * Wrap `middleware`.
 * Can be sync or async; return a promise, receive a callback, or return new
 * values and errors.
 *
 * @param {Middleware} middleware
 * @param {Callback} callback
 */
export function wrap(
  middleware: Middleware,
  callback: Callback
): (...parameters: Array<any>) => void
export type Callback = (
  error?: Error | null | undefined,
  ...output: Array<any>
) => void
export type Middleware = (...input: Array<any>) => any
/**
 * Call all middleware.
 */
export type Run = (...input: Array<any>) => void
/**
 * Add `fn` (middleware) to the list.
 */
export type Use = (fn: Middleware) => Pipeline
/**
 * Middleware.
 */
export type Pipeline = {
  run: Run
  use: Use
}
