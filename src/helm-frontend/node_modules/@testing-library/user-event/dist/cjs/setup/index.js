'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

var setup = require('./setup.js');
var directApi = require('./directApi.js');

const userEvent = {
    ...directApi,
    setup: setup.setupMain
};

exports.userEvent = userEvent;
