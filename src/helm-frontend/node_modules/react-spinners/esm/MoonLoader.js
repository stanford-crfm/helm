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
var moon = createAnimation("MoonLoader", "100% {transform: rotate(360deg)}", "moon");
function MoonLoader(_a) {
    var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.size, size = _f === void 0 ? 60 : _f, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "size"]);
    var _g = parseLengthAndUnit(size), value = _g.value, unit = _g.unit;
    var moonSize = value / 7;
    var wrapper = __assign({ display: "inherit", position: "relative", width: "".concat("".concat(value + moonSize * 2).concat(unit)), height: "".concat("".concat(value + moonSize * 2).concat(unit)), animation: "".concat(moon, " ").concat(0.6 / speedMultiplier, "s 0s infinite linear"), animationFillMode: "forwards" }, cssOverride);
    var ballStyle = function (size) {
        return {
            width: cssValue(size),
            height: cssValue(size),
            borderRadius: "100%",
        };
    };
    var ball = __assign(__assign({}, ballStyle(moonSize)), { backgroundColor: "".concat(color), opacity: "0.8", position: "absolute", top: "".concat("".concat(value / 2 - moonSize / 2).concat(unit)), animation: "".concat(moon, " ").concat(0.6 / speedMultiplier, "s 0s infinite linear"), animationFillMode: "forwards" });
    var circle = __assign(__assign({}, ballStyle(value)), { border: "".concat(moonSize, "px solid ").concat(color), opacity: "0.1", boxSizing: "content-box", position: "absolute" });
    if (!loading) {
        return null;
    }
    return (React.createElement("span", __assign({ style: wrapper }, additionalprops),
        React.createElement("span", { style: ball }),
        React.createElement("span", { style: circle })));
}
export default MoonLoader;
