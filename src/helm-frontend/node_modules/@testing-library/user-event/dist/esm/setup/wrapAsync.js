import * as named from '@testing-library/dom';

const { getConfig } = named;
/**
 * Wrap an internal Promise
 */ function wrapAsync(implementation) {
    return getConfig().asyncWrapper(implementation);
}

export { wrapAsync };
