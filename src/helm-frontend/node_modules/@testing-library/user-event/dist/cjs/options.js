'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

exports.PointerEventsCheckLevel = void 0;
(function(PointerEventsCheckLevel) {
    PointerEventsCheckLevel[PointerEventsCheckLevel[/**
   * Check pointer events on every user interaction that triggers a bunch of events.
   * E.g. once for releasing a mouse button even though this triggers `pointerup`, `mouseup`, `click`, etc...
   */ "EachTrigger"] = 4] = "EachTrigger";
    PointerEventsCheckLevel[PointerEventsCheckLevel[/** Check each target once per call to pointer (related) API */ "EachApiCall"] = 2] = "EachApiCall";
    PointerEventsCheckLevel[PointerEventsCheckLevel[/** Check each event target once */ "EachTarget"] = 1] = "EachTarget";
    PointerEventsCheckLevel[PointerEventsCheckLevel[/** No pointer events check */ "Never"] = 0] = "Never";
})(exports.PointerEventsCheckLevel || (exports.PointerEventsCheckLevel = {}));
