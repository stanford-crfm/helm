'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var selector = require('./selector.js');

function isFocusable(element) {
    return element.matches(selector.FOCUSABLE_SELECTOR);
}

exports.isFocusable = isFocusable;
