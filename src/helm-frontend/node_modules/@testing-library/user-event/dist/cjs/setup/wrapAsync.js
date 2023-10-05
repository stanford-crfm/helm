'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var named = require('@testing-library/dom');

function _interopNamespace(e) {
  if (e && e.__esModule) return e;
  var n = Object.create(null);
  if (e) {
    Object.keys(e).forEach(function (k) {
      if (k !== 'default') {
        var d = Object.getOwnPropertyDescriptor(e, k);
        Object.defineProperty(n, k, d.get ? d : {
          enumerable: true,
          get: function () { return e[k]; }
        });
      }
    });
  }
  n["default"] = e;
  return Object.freeze(n);
}

var named__namespace = /*#__PURE__*/_interopNamespace(named);

const { getConfig } = named__namespace;
/**
 * Wrap an internal Promise
 */ function wrapAsync(implementation) {
    return getConfig().asyncWrapper(implementation);
}

exports.wrapAsync = wrapAsync;
