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
    exports.calculateRgba = void 0;
    var BasicColors;
    (function (BasicColors) {
        BasicColors["maroon"] = "#800000";
        BasicColors["red"] = "#FF0000";
        BasicColors["orange"] = "#FFA500";
        BasicColors["yellow"] = "#FFFF00";
        BasicColors["olive"] = "#808000";
        BasicColors["green"] = "#008000";
        BasicColors["purple"] = "#800080";
        BasicColors["fuchsia"] = "#FF00FF";
        BasicColors["lime"] = "#00FF00";
        BasicColors["teal"] = "#008080";
        BasicColors["aqua"] = "#00FFFF";
        BasicColors["blue"] = "#0000FF";
        BasicColors["navy"] = "#000080";
        BasicColors["black"] = "#000000";
        BasicColors["gray"] = "#808080";
        BasicColors["silver"] = "#C0C0C0";
        BasicColors["white"] = "#FFFFFF";
    })(BasicColors || (BasicColors = {}));
    var calculateRgba = function (color, opacity) {
        if (Object.keys(BasicColors).includes(color)) {
            color = BasicColors[color];
        }
        if (color[0] === "#") {
            color = color.slice(1);
        }
        if (color.length === 3) {
            var res_1 = "";
            color.split("").forEach(function (c) {
                res_1 += c;
                res_1 += c;
            });
            color = res_1;
        }
        var rgbValues = (color.match(/.{2}/g) || []).map(function (hex) { return parseInt(hex, 16); }).join(", ");
        return "rgba(".concat(rgbValues, ", ").concat(opacity, ")");
    };
    exports.calculateRgba = calculateRgba;
});
