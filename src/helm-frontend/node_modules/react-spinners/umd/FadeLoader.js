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
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
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
(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define(["require", "exports", "react", "./helpers/unitConverter", "./helpers/animation"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var React = __importStar(require("react"));
    var unitConverter_1 = require("./helpers/unitConverter");
    var animation_1 = require("./helpers/animation");
    var fade = (0, animation_1.createAnimation)("FadeLoader", "50% {opacity: 0.3} 100% {opacity: 1}", "fade");
    function FadeLoader(_a) {
        var _b = _a.loading, loading = _b === void 0 ? true : _b, _c = _a.color, color = _c === void 0 ? "#000000" : _c, _d = _a.speedMultiplier, speedMultiplier = _d === void 0 ? 1 : _d, _e = _a.cssOverride, cssOverride = _e === void 0 ? {} : _e, _f = _a.height, height = _f === void 0 ? 15 : _f, _g = _a.width, width = _g === void 0 ? 5 : _g, _h = _a.radius, radius = _h === void 0 ? 2 : _h, _j = _a.margin, margin = _j === void 0 ? 2 : _j, additionalprops = __rest(_a, ["loading", "color", "speedMultiplier", "cssOverride", "height", "width", "radius", "margin"]);
        var value = (0, unitConverter_1.parseLengthAndUnit)(margin).value;
        var radiusValue = value + 18;
        var quarter = radiusValue / 2 + radiusValue / 5.5;
        var wrapper = __assign({ display: "inherit", position: "relative", fontSize: "0", top: radiusValue, left: radiusValue, width: "".concat(radiusValue * 3, "px"), height: "".concat(radiusValue * 3, "px") }, cssOverride);
        var style = function (i) {
            return {
                position: "absolute",
                width: (0, unitConverter_1.cssValue)(width),
                height: (0, unitConverter_1.cssValue)(height),
                margin: (0, unitConverter_1.cssValue)(margin),
                backgroundColor: color,
                borderRadius: (0, unitConverter_1.cssValue)(radius),
                transition: "2s",
                animationFillMode: "both",
                animation: "".concat(fade, " ").concat(1.2 / speedMultiplier, "s ").concat(i * 0.12, "s infinite ease-in-out"),
            };
        };
        var a = __assign(__assign({}, style(1)), { top: "".concat(radiusValue, "px"), left: "0" });
        var b = __assign(__assign({}, style(2)), { top: "".concat(quarter, "px"), left: "".concat(quarter, "px"), transform: "rotate(-45deg)" });
        var c = __assign(__assign({}, style(3)), { top: "0", left: "".concat(radiusValue, "px"), transform: "rotate(90deg)" });
        var d = __assign(__assign({}, style(4)), { top: "".concat(-1 * quarter, "px"), left: "".concat(quarter, "px"), transform: "rotate(45deg)" });
        var e = __assign(__assign({}, style(5)), { top: "".concat(-1 * radiusValue, "px"), left: "0" });
        var f = __assign(__assign({}, style(6)), { top: "".concat(-1 * quarter, "px"), left: "".concat(-1 * quarter, "px"), transform: "rotate(-45deg)" });
        var g = __assign(__assign({}, style(7)), { top: "0", left: "".concat(-1 * radiusValue, "px"), transform: "rotate(90deg)" });
        var h = __assign(__assign({}, style(8)), { top: "".concat(quarter, "px"), left: "".concat(-1 * quarter, "px"), transform: "rotate(45deg)" });
        if (!loading) {
            return null;
        }
        return (React.createElement("span", __assign({ style: wrapper }, additionalprops),
            React.createElement("span", { style: a }),
            React.createElement("span", { style: b }),
            React.createElement("span", { style: c }),
            React.createElement("span", { style: d }),
            React.createElement("span", { style: e }),
            React.createElement("span", { style: f }),
            React.createElement("span", { style: g }),
            React.createElement("span", { style: h })));
    }
    exports.default = FadeLoader;
});
