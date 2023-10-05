'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function cloneEvent(event) {
    return new event.constructor(event.type, event);
}

exports.cloneEvent = cloneEvent;
