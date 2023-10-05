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
var bounce = createAnimation("BounceLoader", "0% {transform: scale(0)} 50% {transform: scale(1.0)} 100% {transform: scale(0)}", "bounce");
function BounceLoader(_a) {
    var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.size, size = _f === void 0 ? 60 : _f, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "size"]);
    var style = function (i) {
        var animationTiming = i === 1 ? "".concat(1 / speedMultiplier, "s") : "0s";
        return {
            position: "absolute",
            height: cssValue(size),
            width: cssValue(size),
            backgroundColor: color,
            borderRadius: "100%",
            opacity: 0.6,
            top: 0,
            left: 0,
            animationFillMode: "both",
            animation: "".concat(bounce, " ").concat(2.1 / speedMultiplier, "s ").concat(animationTiming, " infinite ease-in-out"),
        };
    };
    var wrapper = __assign({ display: "inherit", position: "relative", width: cssValue(size), height: cssValue(size) }, cssOverride);
    if (!loading) {
        return null;
    }
    return (React.createElement("span", __assign({ style: wrapper }, additionalprops),
        React.createElement("span", { style: style(1) }),
        React.createElement("span", { style: style(2) })));
}
export default BounceLoader;
