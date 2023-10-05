(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define(["require", "exports", "./unitConverter"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    var unitConverter_1 = require("./unitConverter");
    describe("unitConverter", function () {
        describe("parseLengthAndUnit", function () {
            var spy = jest.spyOn(console, "warn").mockImplementation();
            var output = {
                value: 12,
                unit: "px",
            };
            it("is a function", function () {
                expect(typeof unitConverter_1.parseLengthAndUnit).toEqual("function");
            });
            it("takes a number as the input and append px to the value", function () {
                expect((0, unitConverter_1.parseLengthAndUnit)(12)).toEqual(output);
                expect(spy).not.toBeCalled();
            });
            it("take a string with valid integer css unit and return an object with value and unit", function () {
                expect((0, unitConverter_1.parseLengthAndUnit)("12px")).toEqual(output);
                expect(spy).not.toBeCalled();
            });
            it("take a string with valid css float unit and return an object with value and unit", function () {
                var output = {
                    value: 12.5,
                    unit: "px",
                };
                expect((0, unitConverter_1.parseLengthAndUnit)("12.5px")).toEqual(output);
                expect(spy).not.toBeCalled();
            });
            it("takes an invalid css unit and default the value to px", function () {
                expect((0, unitConverter_1.parseLengthAndUnit)("12fd")).toEqual(output);
                expect(spy).toBeCalled();
            });
        });
        describe("cssValue", function () {
            it("is a function", function () {
                expect(typeof unitConverter_1.cssValue).toEqual("function");
            });
            it("takes a number as the input and append px to the value", function () {
                expect((0, unitConverter_1.cssValue)(12)).toEqual("12px");
            });
            it("takes a string with valid css unit as the input and return the value", function () {
                expect((0, unitConverter_1.cssValue)("12%")).toEqual("12%");
                expect((0, unitConverter_1.cssValue)("12em")).toEqual("12em");
            });
            it("takes a string with invalid css unit as the input and default to px", function () {
                expect((0, unitConverter_1.cssValue)("12qw")).toEqual("12px");
            });
        });
    });
});
