(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define(["require", "exports"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.createAnimation = void 0;
    var createAnimation = function (loaderName, frames, suffix) {
        var animationName = "react-spinners-".concat(loaderName, "-").concat(suffix);
        if (typeof window == "undefined" || !window.document) {
            return animationName;
        }
        var styleEl = document.createElement("style");
        document.head.appendChild(styleEl);
        var styleSheet = styleEl.sheet;
        var keyFrames = "\n    @keyframes ".concat(animationName, " {\n      ").concat(frames, "\n    }\n  ");
        if (styleSheet) {
            styleSheet.insertRule(keyFrames, 0);
        }
        return animationName;
    };
    exports.createAnimation = createAnimation;
});
