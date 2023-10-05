'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function wait(config) {
    const delay = config.delay;
    if (typeof delay !== 'number') {
        return;
    }
    return Promise.all([
        new Promise((resolve)=>globalThis.setTimeout(()=>resolve(), delay)),
        config.advanceTimers(delay)
    ]);
}

exports.wait = wait;
