var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __rest = (this && this.__rest) || function (s, e) {
    var t = {};
    for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
        t[p] = s[p];
    if (s != null && typeof Object.getOwnPropertySymbols === "function")
        for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
            if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
                t[p[i]] = s[p[i]];
        }
    return t;
};
import * as React from "react";
import { cssValue } from "./helpers/unitConverter";
import { createAnimation } from "./helpers/animation";
var square = createAnimation("SquareLoader", "25% {transform: rotateX(180deg) rotateY(0)}\n  50% {transform: rotateX(180deg) rotateY(180deg)} \n  75% {transform: rotateX(0) rotateY(180deg)} \n  100% {transform: rotateX(0) rotateY(0)}", "square");
function SquareLoader(_a) {
    var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.size, size = _f === void 0 ? 50 : _f, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "size"]);
    var style = __assign({ backgroundColor: color, width: cssValue(size), height: cssValue(size), display: "inline-block", animation: "".concat(square, " ").concat(3 / speedMultiplier, "s 0s infinite cubic-bezier(0.09, 0.57, 0.49, 0.9)"), animationFillMode: "both" }, cssOverride);
    if (!loading) {
        return null;
    }
    return React.createElement("span", __assign({ style: style }, additionalprops));
}
export default SquareLoader;
