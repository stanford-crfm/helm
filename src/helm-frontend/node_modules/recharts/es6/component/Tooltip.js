import _isNil from "lodash/isNil";
import _isFunction from "lodash/isFunction";
import _uniqBy from "lodash/uniqBy";
function _typeof(obj) { "@babel/helpers - typeof"; return _typeof = "function" == typeof Symbol && "symbol" == typeof Symbol.iterator ? function (obj) { return typeof obj; } : function (obj) { return obj && "function" == typeof Symbol && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }, _typeof(obj); }
function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); enumerableOnly && (symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; })), keys.push.apply(keys, symbols); } return keys; }
function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = null != arguments[i] ? arguments[i] : {}; i % 2 ? ownKeys(Object(source), !0).forEach(function (key) { _defineProperty(target, key, source[key]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)) : ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } return target; }
function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }
function _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, _toPropertyKey(descriptor.key), descriptor); } }
function _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); Object.defineProperty(Constructor, "prototype", { writable: false }); return Constructor; }
function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function"); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, writable: true, configurable: true } }); Object.defineProperty(subClass, "prototype", { writable: false }); if (superClass) _setPrototypeOf(subClass, superClass); }
function _setPrototypeOf(o, p) { _setPrototypeOf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function _setPrototypeOf(o, p) { o.__proto__ = p; return o; }; return _setPrototypeOf(o, p); }
function _createSuper(Derived) { var hasNativeReflectConstruct = _isNativeReflectConstruct(); return function _createSuperInternal() { var Super = _getPrototypeOf(Derived), result; if (hasNativeReflectConstruct) { var NewTarget = _getPrototypeOf(this).constructor; result = Reflect.construct(Super, arguments, NewTarget); } else { result = Super.apply(this, arguments); } return _possibleConstructorReturn(this, result); }; }
function _possibleConstructorReturn(self, call) { if (call && (_typeof(call) === "object" || typeof call === "function")) { return call; } else if (call !== void 0) { throw new TypeError("Derived constructors may only return object or undefined"); } return _assertThisInitialized(self); }
function _assertThisInitialized(self) { if (self === void 0) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return self; }
function _isNativeReflectConstruct() { if (typeof Reflect === "undefined" || !Reflect.construct) return false; if (Reflect.construct.sham) return false; if (typeof Proxy === "function") return true; try { Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {})); return true; } catch (e) { return false; } }
function _getPrototypeOf(o) { _getPrototypeOf = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function _getPrototypeOf(o) { return o.__proto__ || Object.getPrototypeOf(o); }; return _getPrototypeOf(o); }
function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return _typeof(key) === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (_typeof(input) !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (_typeof(res) !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
/**
 * @fileOverview Tooltip
 */
import React, { PureComponent } from 'react';
import { translateStyle } from 'react-smooth';
import classNames from 'classnames';
import { DefaultTooltipContent } from './DefaultTooltipContent';
import { Global } from '../util/Global';
import { isNumber } from '../util/DataUtils';
var CLS_PREFIX = 'recharts-tooltip-wrapper';
var EPS = 1;
function defaultUniqBy(entry) {
  return entry.dataKey;
}
function getUniqPayload(option, payload) {
  if (option === true) {
    return _uniqBy(payload, defaultUniqBy);
  }
  if (_isFunction(option)) {
    return _uniqBy(payload, option);
  }
  return payload;
}
function renderContent(content, props) {
  if ( /*#__PURE__*/React.isValidElement(content)) {
    return /*#__PURE__*/React.cloneElement(content, props);
  }
  if (_isFunction(content)) {
    return /*#__PURE__*/React.createElement(content, props);
  }
  return /*#__PURE__*/React.createElement(DefaultTooltipContent, props);
}
export var Tooltip = /*#__PURE__*/function (_PureComponent) {
  _inherits(Tooltip, _PureComponent);
  var _super = _createSuper(Tooltip);
  function Tooltip() {
    var _this;
    _classCallCheck(this, Tooltip);
    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }
    _this = _super.call.apply(_super, [this].concat(args));
    _defineProperty(_assertThisInitialized(_this), "state", {
      boxWidth: -1,
      boxHeight: -1,
      dismissed: false,
      dismissedAtCoordinate: {
        x: 0,
        y: 0
      }
    });
    _defineProperty(_assertThisInitialized(_this), "handleKeyDown", function (event) {
      if (event.key === 'Escape') {
        _this.setState({
          dismissed: true,
          dismissedAtCoordinate: _objectSpread(_objectSpread({}, _this.state.dismissedAtCoordinate), {}, {
            x: _this.props.coordinate.x,
            y: _this.props.coordinate.y
          })
        });
      }
    });
    _defineProperty(_assertThisInitialized(_this), "getTranslate", function (_ref) {
      var key = _ref.key,
        tooltipDimension = _ref.tooltipDimension,
        viewBoxDimension = _ref.viewBoxDimension;
      var _this$props = _this.props,
        allowEscapeViewBox = _this$props.allowEscapeViewBox,
        reverseDirection = _this$props.reverseDirection,
        coordinate = _this$props.coordinate,
        offset = _this$props.offset,
        position = _this$props.position,
        viewBox = _this$props.viewBox;
      if (position && isNumber(position[key])) {
        return position[key];
      }
      var negative = coordinate[key] - tooltipDimension - offset;
      var positive = coordinate[key] + offset;
      if (allowEscapeViewBox[key]) {
        return reverseDirection[key] ? negative : positive;
      }
      if (reverseDirection[key]) {
        var _tooltipBoundary = negative;
        var _viewBoxBoundary = viewBox[key];
        if (_tooltipBoundary < _viewBoxBoundary) {
          return Math.max(positive, viewBox[key]);
        }
        return Math.max(negative, viewBox[key]);
      }
      var tooltipBoundary = positive + tooltipDimension;
      var viewBoxBoundary = viewBox[key] + viewBoxDimension;
      if (tooltipBoundary > viewBoxBoundary) {
        return Math.max(negative, viewBox[key]);
      }
      return Math.max(positive, viewBox[key]);
    });
    return _this;
  }
  _createClass(Tooltip, [{
    key: "componentDidMount",
    value: function componentDidMount() {
      this.updateBBox();
    }
  }, {
    key: "componentWillUnmount",
    value: function componentWillUnmount() {
      document.removeEventListener('keydown', this.handleKeyDown);
    }
  }, {
    key: "componentDidUpdate",
    value: function componentDidUpdate() {
      this.updateBBox();
    }
  }, {
    key: "updateBBox",
    value: function updateBBox() {
      var _this$state = this.state,
        boxWidth = _this$state.boxWidth,
        boxHeight = _this$state.boxHeight,
        dismissed = _this$state.dismissed;
      if (dismissed) {
        document.removeEventListener('keydown', this.handleKeyDown);
        if (this.props.coordinate.x !== this.state.dismissedAtCoordinate.x || this.props.coordinate.y !== this.state.dismissedAtCoordinate.y) {
          this.setState({
            dismissed: false
          });
        }
      } else {
        document.addEventListener('keydown', this.handleKeyDown);
      }
      if (this.wrapperNode && this.wrapperNode.getBoundingClientRect) {
        var box = this.wrapperNode.getBoundingClientRect();
        if (Math.abs(box.width - boxWidth) > EPS || Math.abs(box.height - boxHeight) > EPS) {
          this.setState({
            boxWidth: box.width,
            boxHeight: box.height
          });
        }
      } else if (boxWidth !== -1 || boxHeight !== -1) {
        this.setState({
          boxWidth: -1,
          boxHeight: -1
        });
      }
    }
  }, {
    key: "render",
    value: function render() {
      var _classNames,
        _this2 = this;
      var _this$props2 = this.props,
        payload = _this$props2.payload,
        isAnimationActive = _this$props2.isAnimationActive,
        animationDuration = _this$props2.animationDuration,
        animationEasing = _this$props2.animationEasing,
        filterNull = _this$props2.filterNull,
        payloadUniqBy = _this$props2.payloadUniqBy;
      var finalPayload = getUniqPayload(payloadUniqBy, filterNull && payload && payload.length ? payload.filter(function (entry) {
        return !_isNil(entry.value);
      }) : payload);
      var hasPayload = finalPayload && finalPayload.length;
      var _this$props3 = this.props,
        content = _this$props3.content,
        viewBox = _this$props3.viewBox,
        coordinate = _this$props3.coordinate,
        position = _this$props3.position,
        active = _this$props3.active,
        wrapperStyle = _this$props3.wrapperStyle;
      var outerStyle = _objectSpread({
        pointerEvents: 'none',
        visibility: !this.state.dismissed && active && hasPayload ? 'visible' : 'hidden',
        position: 'absolute',
        top: 0,
        left: 0
      }, wrapperStyle);
      var translateX, translateY;
      if (position && isNumber(position.x) && isNumber(position.y)) {
        translateX = position.x;
        translateY = position.y;
      } else {
        var _this$state2 = this.state,
          boxWidth = _this$state2.boxWidth,
          boxHeight = _this$state2.boxHeight;
        if (boxWidth > 0 && boxHeight > 0 && coordinate) {
          translateX = this.getTranslate({
            key: 'x',
            tooltipDimension: boxWidth,
            viewBoxDimension: viewBox.width
          });
          translateY = this.getTranslate({
            key: 'y',
            tooltipDimension: boxHeight,
            viewBoxDimension: viewBox.height
          });
        } else {
          outerStyle.visibility = 'hidden';
        }
      }
      outerStyle = _objectSpread(_objectSpread({}, translateStyle({
        transform: this.props.useTranslate3d ? "translate3d(".concat(translateX, "px, ").concat(translateY, "px, 0)") : "translate(".concat(translateX, "px, ").concat(translateY, "px)")
      })), outerStyle);
      if (isAnimationActive && active) {
        outerStyle = _objectSpread(_objectSpread({}, translateStyle({
          transition: "transform ".concat(animationDuration, "ms ").concat(animationEasing)
        })), outerStyle);
      }
      var cls = classNames(CLS_PREFIX, (_classNames = {}, _defineProperty(_classNames, "".concat(CLS_PREFIX, "-right"), isNumber(translateX) && coordinate && isNumber(coordinate.x) && translateX >= coordinate.x), _defineProperty(_classNames, "".concat(CLS_PREFIX, "-left"), isNumber(translateX) && coordinate && isNumber(coordinate.x) && translateX < coordinate.x), _defineProperty(_classNames, "".concat(CLS_PREFIX, "-bottom"), isNumber(translateY) && coordinate && isNumber(coordinate.y) && translateY >= coordinate.y), _defineProperty(_classNames, "".concat(CLS_PREFIX, "-top"), isNumber(translateY) && coordinate && isNumber(coordinate.y) && translateY < coordinate.y), _classNames));
      return (
        /*#__PURE__*/
        // ESLint is disabled to allow listening to the `Escape` key. Refer to
        // https://github.com/recharts/recharts/pull/2925
        // eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions
        React.createElement("div", {
          tabIndex: -1,
          role: "dialog",
          className: cls,
          style: outerStyle,
          ref: function ref(node) {
            _this2.wrapperNode = node;
          }
        }, renderContent(content, _objectSpread(_objectSpread({}, this.props), {}, {
          payload: finalPayload
        })))
      );
    }
  }]);
  return Tooltip;
}(PureComponent);
_defineProperty(Tooltip, "displayName", 'Tooltip');
_defineProperty(Tooltip, "defaultProps", {
  active: false,
  allowEscapeViewBox: {
    x: false,
    y: false
  },
  reverseDirection: {
    x: false,
    y: false
  },
  offset: 10,
  viewBox: {
    x: 0,
    y: 0,
    height: 0,
    width: 0
  },
  coordinate: {
    x: 0,
    y: 0
  },
  cursorStyle: {},
  separator: ' : ',
  wrapperStyle: {},
  contentStyle: {},
  itemStyle: {},
  labelStyle: {},
  cursor: true,
  trigger: 'hover',
  isAnimationActive: !Global.isSsr,
  animationEasing: 'ease',
  animationDuration: 400,
  filterNull: true,
  useTranslate3d: false
});