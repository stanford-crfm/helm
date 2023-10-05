"use strict";
/**
 * @jest-environment node
 */
Object.defineProperty(exports, "__esModule", { value: true });
var animation_1 = require("./animation");
describe("animation", function () {
    it("should not throw an error on server side", function () {
        var name = (0, animation_1.createAnimation)("TestLoader", "0% {left: -35%;right: 100%} 60% {left: 100%;right: -90%} 100% {left: 100%;right: -90%}", "my-suffix");
        expect(name).toEqual("react-spinners-TestLoader-my-suffix");
    });
});
