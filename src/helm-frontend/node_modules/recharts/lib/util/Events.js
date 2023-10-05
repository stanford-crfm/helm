"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.eventCenter = exports.SYNC_EVENT = void 0;
var _eventemitter = _interopRequireDefault(require("eventemitter3"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
var eventCenter = new _eventemitter["default"]();
exports.eventCenter = eventCenter;
if (eventCenter.setMaxListeners) {
  eventCenter.setMaxListeners(10);
}
var SYNC_EVENT = 'recharts.syncMouseEvents';
exports.SYNC_EVENT = SYNC_EVENT;