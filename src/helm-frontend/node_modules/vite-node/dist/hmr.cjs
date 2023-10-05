'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var hmr = require('./chunk-hmr.cjs');
require('node:events');
require('picocolors');
require('debug');



exports.createHmrEmitter = hmr.createHmrEmitter;
exports.createHotContext = hmr.createHotContext;
exports.getCache = hmr.getCache;
exports.handleMessage = hmr.handleMessage;
exports.reload = hmr.reload;
exports.sendMessageBuffer = hmr.sendMessageBuffer;
exports.viteNodeHmrPlugin = hmr.viteNodeHmrPlugin;
