'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function getDocumentFromNode(el) {
    return isDocument(el) ? el : el.ownerDocument;
}
function isDocument(node) {
    return node.nodeType === 9;
}

exports.getDocumentFromNode = getDocumentFromNode;
