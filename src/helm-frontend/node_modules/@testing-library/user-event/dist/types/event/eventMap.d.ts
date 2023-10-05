import { EventType } from './types';
export declare const eventMap: {
    readonly click: {
        readonly EventType: "PointerEvent";
        readonly defaultInit: {
            readonly bubbles: true;
            readonly cancelable: true;
            readonly composed: true;
        };
    };
    readonly auxclick: {
        readonly EventType: "PointerEvent";
        readonly defaultInit: {
            readonly bubbles: true;
            readonly cancelable: true;
            readonly composed: true;
        };
    };
    readonly contextmenu: {
        readonly EventType: "PointerEvent";
        readonly defaultInit: {
            readonly bubbles: true;
            readonly cancelable: true;
            readonly composed: true;
        };
    };
    readonly beforeInput: {
        readonly EventType: "InputEvent";
        readonly defaultInit: {
            readonly bubbles: true;
            readonly cancelable: true;
            readonly composed: true;
        };
    };
    readonly input: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly progress: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly select: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly scroll: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly copy: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly blur: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly focus: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly cut: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly paste: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly abort: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly change: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly drag: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly drop: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly emptied: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly ended: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly error: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly invalid: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly load: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pause: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly play: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly playing: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly reset: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly resize: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly seeked: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly seeking: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly stalled: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly submit: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly suspend: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly waiting: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly wheel: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly compositionEnd: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly compositionStart: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly compositionUpdate: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly keyDown: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly keyPress: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly keyUp: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly focusIn: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly focusOut: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly contextMenu: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dblClick: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dragEnd: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dragEnter: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dragExit: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dragLeave: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dragOver: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly dragStart: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseDown: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseEnter: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseLeave: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseMove: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseOut: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseOver: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly mouseUp: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly popState: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly touchCancel: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly touchEnd: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly touchMove: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly touchStart: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly canPlay: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly canPlayThrough: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly durationChange: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly encrypted: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly loadedData: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly loadedMetadata: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly loadStart: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly rateChange: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly timeUpdate: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly volumeChange: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly animationStart: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly animationEnd: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly animationIteration: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly transitionCancel: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly transitionEnd: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly transitionRun: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly transitionStart: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly doubleClick: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerOver: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerEnter: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerDown: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerMove: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerUp: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerCancel: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerOut: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly pointerLeave: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly gotPointerCapture: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly lostPointerCapture: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly offline: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
    readonly online: {
        EventType: EventInterface;
        defaultInit: EventInit;
    };
};
export declare const eventMapKeys: {
    [k in keyof DocumentEventMap]?: keyof typeof eventMap;
};
export declare function isMouseEvent(type: EventType): boolean;
export declare function isKeyboardEvent(type: EventType): boolean;
