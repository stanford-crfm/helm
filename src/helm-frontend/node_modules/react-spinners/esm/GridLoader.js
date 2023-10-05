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
import { cssValue, parseLengthAndUnit } from "./helpers/unitConverter";
import { createAnimation } from "./helpers/animation";
var grid = createAnimation("GridLoader", "0% {transform: scale(1)} 50% {transform: scale(0.5); opacity: 0.7} 100% {transform: scale(1); opacity: 1}", "grid");
var random = function (top) { return Math.random() * top; };
function GridLoader(_a) {
    var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.size, size = _f === void 0 ? 15 : _f, _g = _a.margin, margin = _g === void 0 ? 2 : _g, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "size", "margin"]);
    var sizeWithUnit = parseLengthAndUnit(size);
    var marginWithUnit = parseLengthAndUnit(margin);
    var width = parseFloat(sizeWithUnit.value.toString()) * 3 + parseFloat(marginWithUnit.value.toString()) * 6;
    var wrapper = __assign({ width: "".concat(width).concat(sizeWithUnit.unit), fontSize: 0, display: "inline-block" }, cssOverride);
    var style = function (rand) {
        return {
            display: "inline-block",
            backgroundColor: color,
            width: "".concat(cssValue(size)),
            height: "".concat(cssValue(size)),
            margin: cssValue(margin),
            borderRadius: "100%",
            animationFillMode: "both",
            animation: "".concat(grid, " ").concat((rand / 100 + 0.6) / speedMultiplier, "s ").concat(rand / 100 - 0.2, "s infinite ease"),
        };
    };
    if (!loading) {
        return null;
    }
    return (React.createElement("span", __assign({ style: wrapper }, additionalprops, { ref: function (node) {
            if (node) {
                node.style.setProperty("width", "".concat(width).concat(sizeWithUnit.unit), "important");
            }
        } }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) }),
        React.createElement("span", { style: style(random(100)) })));
}
export default GridLoader;
