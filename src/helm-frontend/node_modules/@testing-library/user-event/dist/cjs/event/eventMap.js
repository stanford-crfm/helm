'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var named = require('@testing-library/dom/dist/event-map.js');

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

const eventMap = {
    ...named__namespace.eventMap,
    click: {
        EventType: 'PointerEvent',
        defaultInit: {
            bubbles: true,
            cancelable: true,
            composed: true
        }
    },
    auxclick: {
        EventType: 'PointerEvent',
        defaultInit: {
            bubbles: true,
            cancelable: true,
            composed: true
        }
    },
    contextmenu: {
        EventType: 'PointerEvent',
        defaultInit: {
            bubbles: true,
            cancelable: true,
            composed: true
        }
    },
    beforeInput: {
        EventType: 'InputEvent',
        defaultInit: {
            bubbles: true,
            cancelable: true,
            composed: true
        }
    }
};
const eventMapKeys = Object.fromEntries(Object.keys(eventMap).map((k)=>[
        k.toLowerCase(),
        k
    ]));
function getEventClass(type) {
    const k = eventMapKeys[type];
    return k && eventMap[k].EventType;
}
const mouseEvents = [
    'MouseEvent',
    'PointerEvent'
];
function isMouseEvent(type) {
    return mouseEvents.includes(getEventClass(type));
}
function isKeyboardEvent(type) {
    return getEventClass(type) === 'KeyboardEvent';
}

exports.eventMap = eventMap;
exports.eventMapKeys = eventMapKeys;
exports.isKeyboardEvent = isKeyboardEvent;
exports.isMouseEvent = isMouseEvent;
