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
var climbingBox = createAnimation("ClimbingBoxLoader", "0% {transform:translate(0, -1em) rotate(-45deg)}\n  5% {transform:translate(0, -1em) rotate(-50deg)}\n  20% {transform:translate(1em, -2em) rotate(47deg)}\n  25% {transform:translate(1em, -2em) rotate(45deg)}\n  30% {transform:translate(1em, -2em) rotate(40deg)}\n  45% {transform:translate(2em, -3em) rotate(137deg)}\n  50% {transform:translate(2em, -3em) rotate(135deg)}\n  55% {transform:translate(2em, -3em) rotate(130deg)}\n  70% {transform:translate(3em, -4em) rotate(217deg)}\n  75% {transform:translate(3em, -4em) rotate(220deg)}\n  100% {transform:translate(0, -1em) rotate(-225deg)}", "climbingBox");
function ClimbingBoxLoader(_a) {
    var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.size, size = _f === void 0 ? 15 : _f, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "size"]);
    var container = __assign({ display: "inherit", position: "relative", width: "7.1em", height: "7.1em" }, cssOverride);
    var wrapper = {
        position: "absolute",
        top: "50%",
        left: "50%",
        marginTop: "-2.7em",
        marginLeft: "-2.7em",
        width: "5.4em",
        height: "5.4em",
        fontSize: cssValue(size),
    };
    var style = {
        position: "absolute",
        left: "0",
        bottom: "-0.1em",
        height: "1em",
        width: "1em",
        backgroundColor: "transparent",
        borderRadius: "15%",
        border: "0.25em solid ".concat(color),
        transform: "translate(0, -1em) rotate(-45deg)",
        animationFillMode: "both",
        animation: "".concat(climbingBox, " ").concat(2.5 / speedMultiplier, "s infinite cubic-bezier(0.79, 0, 0.47, 0.97)"),
    };
    var hill = {
        position: "absolute",
        width: "7.1em",
        height: "7.1em",
        top: "1.7em",
        left: "1.7em",
        borderLeft: "0.25em solid ".concat(color),
        transform: "rotate(45deg)",
    };
    if (!loading) {
        return null;
    }
    return (React.createElement("span", __assign({ style: container }, additionalprops),
        React.createElement("span", { style: wrapper },
            React.createElement("span", { style: style }),
            React.createElement("span", { style: hill }))));
}
export default ClimbingBoxLoader;
