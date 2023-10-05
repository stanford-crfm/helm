"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.findEntryInArray = findEntryInArray;
exports.uniqueId = exports.mathSign = exports.isPercent = exports.isNumber = exports.isNumOrStr = exports.interpolateNumber = exports.hasDuplicate = exports.getPercentValue = exports.getLinearRegression = exports.getAnyElementOfObject = void 0;
var _get2 = _interopRequireDefault(require("lodash/get"));
var _isArray2 = _interopRequireDefault(require("lodash/isArray"));
var _isNaN2 = _interopRequireDefault(require("lodash/isNaN"));
var _isNumber2 = _interopRequireDefault(require("lodash/isNumber"));
var _isString2 = _interopRequireDefault(require("lodash/isString"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { "default": obj }; }
var mathSign = function mathSign(value) {
  if (value === 0) {
    return 0;
  }
  if (value > 0) {
    return 1;
  }
  return -1;
};
exports.mathSign = mathSign;
var isPercent = function isPercent(value) {
  return (0, _isString2["default"])(value) && value.indexOf('%') === value.length - 1;
};
exports.isPercent = isPercent;
var isNumber = function isNumber(value) {
  return (0, _isNumber2["default"])(value) && !(0, _isNaN2["default"])(value);
};
exports.isNumber = isNumber;
var isNumOrStr = function isNumOrStr(value) {
  return isNumber(value) || (0, _isString2["default"])(value);
};
exports.isNumOrStr = isNumOrStr;
var idCounter = 0;
var uniqueId = function uniqueId(prefix) {
  var id = ++idCounter;
  return "".concat(prefix || '').concat(id);
};

/**
 * Get percent value of a total value
 * @param {number|string} percent A percent
 * @param {number} totalValue     Total value
 * @param {number} defaultValue   The value returned when percent is undefined or invalid
 * @param {boolean} validate      If set to be true, the result will be validated
 * @return {number} value
 */
exports.uniqueId = uniqueId;
var getPercentValue = function getPercentValue(percent, totalValue) {
  var defaultValue = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 0;
  var validate = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : false;
  if (!isNumber(percent) && !(0, _isString2["default"])(percent)) {
    return defaultValue;
  }
  var value;
  if (isPercent(percent)) {
    var index = percent.indexOf('%');
    value = totalValue * parseFloat(percent.slice(0, index)) / 100;
  } else {
    value = +percent;
  }
  if ((0, _isNaN2["default"])(value)) {
    value = defaultValue;
  }
  if (validate && value > totalValue) {
    value = totalValue;
  }
  return value;
};
exports.getPercentValue = getPercentValue;
var getAnyElementOfObject = function getAnyElementOfObject(obj) {
  if (!obj) {
    return null;
  }
  var keys = Object.keys(obj);
  if (keys && keys.length) {
    return obj[keys[0]];
  }
  return null;
};
exports.getAnyElementOfObject = getAnyElementOfObject;
var hasDuplicate = function hasDuplicate(ary) {
  if (!(0, _isArray2["default"])(ary)) {
    return false;
  }
  var len = ary.length;
  var cache = {};
  for (var i = 0; i < len; i++) {
    if (!cache[ary[i]]) {
      cache[ary[i]] = true;
    } else {
      return true;
    }
  }
  return false;
};

/* @todo consider to rename this function into `getInterpolator` */
exports.hasDuplicate = hasDuplicate;
var interpolateNumber = function interpolateNumber(numberA, numberB) {
  if (isNumber(numberA) && isNumber(numberB)) {
    return function (t) {
      return numberA + t * (numberB - numberA);
    };
  }
  return function () {
    return numberB;
  };
};
exports.interpolateNumber = interpolateNumber;
function findEntryInArray(ary, specifiedKey, specifiedValue) {
  if (!ary || !ary.length) {
    return null;
  }
  return ary.find(function (entry) {
    return entry && (typeof specifiedKey === 'function' ? specifiedKey(entry) : (0, _get2["default"])(entry, specifiedKey)) === specifiedValue;
  });
}

/**
 * The least square linear regression
 * @param {Array} data The array of points
 * @returns {Object} The domain of x, and the parameter of linear function
 */
var getLinearRegression = function getLinearRegression(data) {
  if (!data || !data.length) {
    return null;
  }
  var len = data.length;
  var xsum = 0;
  var ysum = 0;
  var xysum = 0;
  var xxsum = 0;
  var xmin = Infinity;
  var xmax = -Infinity;
  var xcurrent = 0;
  var ycurrent = 0;
  for (var i = 0; i < len; i++) {
    xcurrent = data[i].cx || 0;
    ycurrent = data[i].cy || 0;
    xsum += xcurrent;
    ysum += ycurrent;
    xysum += xcurrent * ycurrent;
    xxsum += xcurrent * xcurrent;
    xmin = Math.min(xmin, xcurrent);
    xmax = Math.max(xmax, xcurrent);
  }
  var a = len * xxsum !== xsum * xsum ? (len * xysum - xsum * ysum) / (len * xxsum - xsum * xsum) : 0;
  return {
    xmin: xmin,
    xmax: xmax,
    a: a,
    b: (ysum - a * xsum) / len
  };
};
exports.getLinearRegression = getLinearRegression;