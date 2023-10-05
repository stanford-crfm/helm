import * as named from '@testing-library/dom/dist/event-map.js';

const eventMap = {
    ...named.eventMap,
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

export { eventMap, eventMapKeys, isKeyboardEvent, isMouseEvent };
