function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }
function _slicedToArray(arr, i) { return _arrayWithHoles(arr) || _iterableToArrayLimit(arr, i) || _unsupportedIterableToArray(arr, i) || _nonIterableRest(); }
function _nonIterableRest() { throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }
function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }
function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i]; return arr2; }
function _iterableToArrayLimit(arr, i) { var _i = null == arr ? null : "undefined" != typeof Symbol && arr[Symbol.iterator] || arr["@@iterator"]; if (null != _i) { var _s, _e, _x, _r, _arr = [], _n = !0, _d = !1; try { if (_x = (_i = _i.call(arr)).next, 0 === i) { if (Object(_i) !== _i) return; _n = !1; } else for (; !(_n = (_s = _x.call(_i)).done) && (_arr.push(_s.value), _arr.length !== i); _n = !0); } catch (err) { _d = !0, _e = err; } finally { try { if (!_n && null != _i["return"] && (_r = _i["return"](), Object(_r) !== _r)) return; } finally { if (_d) throw _e; } } return _arr; } }
function _arrayWithHoles(arr) { if (Array.isArray(arr)) return arr; }
/**
 * @fileOverview Wrapper component to make charts adapt to the size of parent * DOM
 */
import classNames from 'classnames';
import React, { forwardRef, cloneElement, useState, useImperativeHandle, useRef, useEffect, useCallback, useMemo } from 'react';
import ReactResizeDetector from 'react-resize-detector';
import { isPercent } from '../util/DataUtils';
import { warn } from '../util/LogUtils';
export var ResponsiveContainer = /*#__PURE__*/forwardRef(function (_ref, ref) {
  var aspect = _ref.aspect,
    _ref$initialDimension = _ref.initialDimension,
    initialDimension = _ref$initialDimension === void 0 ? {
      width: -1,
      height: -1
    } : _ref$initialDimension,
    _ref$width = _ref.width,
    width = _ref$width === void 0 ? '100%' : _ref$width,
    _ref$height = _ref.height,
    height = _ref$height === void 0 ? '100%' : _ref$height,
    _ref$minWidth = _ref.minWidth,
    minWidth = _ref$minWidth === void 0 ? 0 : _ref$minWidth,
    minHeight = _ref.minHeight,
    maxHeight = _ref.maxHeight,
    children = _ref.children,
    _ref$debounce = _ref.debounce,
    debounce = _ref$debounce === void 0 ? 0 : _ref$debounce,
    id = _ref.id,
    className = _ref.className,
    onResize = _ref.onResize;
  var _useState = useState({
      containerWidth: initialDimension.width,
      containerHeight: initialDimension.height
    }),
    _useState2 = _slicedToArray(_useState, 2),
    sizes = _useState2[0],
    setSizes = _useState2[1];
  var containerRef = useRef(null);
  useImperativeHandle(ref, function () {
    return containerRef;
  }, [containerRef]);
  var getContainerSize = useCallback(function () {
    if (!containerRef.current) {
      return null;
    }
    return {
      containerWidth: containerRef.current.clientWidth,
      containerHeight: containerRef.current.clientHeight
    };
  }, []);
  var updateDimensionsImmediate = useCallback(function () {
    var newSize = getContainerSize();
    if (newSize) {
      var containerWidth = newSize.containerWidth,
        containerHeight = newSize.containerHeight;
      if (onResize) onResize(containerWidth, containerHeight);
      setSizes(function (currentSizes) {
        var oldWidth = currentSizes.containerWidth,
          oldHeight = currentSizes.containerHeight;
        if (containerWidth !== oldWidth || containerHeight !== oldHeight) {
          return {
            containerWidth: containerWidth,
            containerHeight: containerHeight
          };
        }
        return currentSizes;
      });
    }
  }, [getContainerSize, onResize]);
  var chartContent = useMemo(function () {
    var containerWidth = sizes.containerWidth,
      containerHeight = sizes.containerHeight;
    if (containerWidth < 0 || containerHeight < 0) {
      return null;
    }
    warn(isPercent(width) || isPercent(height), "The width(%s) and height(%s) are both fixed numbers,\n       maybe you don't need to use a ResponsiveContainer.", width, height);
    warn(!aspect || aspect > 0, 'The aspect(%s) must be greater than zero.', aspect);
    var calculatedWidth = isPercent(width) ? containerWidth : width;
    var calculatedHeight = isPercent(height) ? containerHeight : height;
    if (aspect && aspect > 0) {
      // Preserve the desired aspect ratio
      if (calculatedWidth) {
        // Will default to using width for aspect ratio
        calculatedHeight = calculatedWidth / aspect;
      } else if (calculatedHeight) {
        // But we should also take height into consideration
        calculatedWidth = calculatedHeight * aspect;
      }

      // if maxHeight is set, overwrite if calculatedHeight is greater than maxHeight
      if (maxHeight && calculatedHeight > maxHeight) {
        calculatedHeight = maxHeight;
      }
    }
    warn(calculatedWidth > 0 || calculatedHeight > 0, "The width(%s) and height(%s) of chart should be greater than 0,\n       please check the style of container, or the props width(%s) and height(%s),\n       or add a minWidth(%s) or minHeight(%s) or use aspect(%s) to control the\n       height and width.", calculatedWidth, calculatedHeight, width, height, minWidth, minHeight, aspect);
    return /*#__PURE__*/cloneElement(children, {
      width: calculatedWidth,
      height: calculatedHeight
    });
  }, [aspect, children, height, maxHeight, minHeight, minWidth, sizes, width]);
  useEffect(function () {
    var size = getContainerSize();
    if (size) {
      setSizes(size);
    }
  }, [getContainerSize]);
  var style = {
    width: width,
    height: height,
    minWidth: minWidth,
    minHeight: minHeight,
    maxHeight: maxHeight
  };
  return /*#__PURE__*/React.createElement(ReactResizeDetector, {
    handleWidth: true,
    handleHeight: true,
    onResize: updateDimensionsImmediate,
    targetRef: containerRef,
    refreshMode: debounce > 0 ? 'debounce' : undefined,
    refreshRate: debounce
  }, /*#__PURE__*/React.createElement("div", _extends({}, id != null ? {
    id: "".concat(id)
  } : {}, {
    className: classNames('recharts-responsive-container', className),
    style: style,
    ref: containerRef
  }), chartContent));
});