import * as named from '@testing-library/dom/dist/helpers.js';

const { getWindowFromNode } = named;
function getWindow(node) {
    return getWindowFromNode(node);
}

export { getWindow };
