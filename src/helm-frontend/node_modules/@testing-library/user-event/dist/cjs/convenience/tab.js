'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

async function tab({ shift } = {}) {
    return this.keyboard(shift === true ? '{Shift>}{Tab}{/Shift}' : shift === false ? '[/ShiftLeft][/ShiftRight]{Tab}' : '{Tab}');
}

exports.tab = tab;
