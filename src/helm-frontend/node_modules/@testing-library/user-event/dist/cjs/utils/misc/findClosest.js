'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function findClosest(element, callback) {
    let el = element;
    do {
        if (callback(el)) {
            return el;
        }
        el = el.parentElement;
    }while (el && el !== element.ownerDocument.body)
    return undefined;
}

exports.findClosest = findClosest;
