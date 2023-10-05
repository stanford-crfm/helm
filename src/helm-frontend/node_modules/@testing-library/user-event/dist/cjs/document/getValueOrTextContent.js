'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

require('../utils/click/isClickableInput.js');
require('../utils/dataTransfer/Clipboard.js');
var isContentEditable = require('../utils/edit/isContentEditable.js');
require('../utils/edit/isEditable.js');
require('../utils/edit/maxLength.js');
require('@testing-library/dom/dist/helpers.js');
require('../utils/keyDef/readNextDescriptor.js');
require('../utils/misc/level.js');
require('../options.js');
var UI = require('./UI.js');

function getValueOrTextContent(element) {
    // istanbul ignore if
    if (!element) {
        return null;
    }
    if (isContentEditable.isContentEditable(element)) {
        return element.textContent;
    }
    return UI.getUIValue(element);
}

exports.getValueOrTextContent = getValueOrTextContent;
