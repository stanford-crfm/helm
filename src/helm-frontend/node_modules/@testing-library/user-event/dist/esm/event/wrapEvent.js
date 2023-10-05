import * as named from '@testing-library/dom';

const { getConfig } = named;
function wrapEvent(cb, _element) {
    return getConfig().eventWrapper(cb);
}

export { wrapEvent };
