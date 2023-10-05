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
import { parseLengthAndUnit, cssValue } from "./helpers/unitConverter";
import { createAnimation } from "./helpers/animation";
var right = createAnimation("RingLoader", "0% {transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg)} 100% {transform: rotateX(180deg) rotateY(360deg) rotateZ(360deg)}", "right");
var left = createAnimation("RingLoader", "0% {transform: rotateX(0deg) rotateY(0deg) rotateZ(0deg)} 100% {transform: rotateX(360deg) rotateY(180deg) rotateZ(360deg)}", "left");
function RingLoader(_a) {
    var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.size, size = _f === void 0 ? 60 : _f, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "size"]);
    var _g = parseLengthAndUnit(size), value = _g.value, unit = _g.unit;
    var wrapper = __assign({ display: "inherit", width: cssValue(size), height: cssValue(size), position: "relative" }, cssOverride);
    var style = function (i) {
        return {
            position: "absolute",
            top: "0",
            left: "0",
            width: "".concat(value).concat(unit),
            height: "".concat(value).concat(unit),
            border: "".concat(value / 10).concat(unit, " solid ").concat(color),
            opacity: "0.4",
            borderRadius: "100%",
            animationFillMode: "forwards",
            perspective: "800px",
            animation: "".concat(i === 1 ? right : left, " ").concat(2 / speedMultiplier, "s 0s infinite linear"),
        };
    };
    if (!loading) {
        return null;
    }
    return (React.createElement("span", __assign({ style: wrapper }, additionalprops),
        React.createElement("span", { style: style(1) }),
        React.createElement("span", { style: style(2) })));
}
export default RingLoader;
