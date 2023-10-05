# Handle element resizes like it's 2023!

<img src="https://img.shields.io/npm/dm/react-resize-detector?style=flat-square"> <img src="https://badgen.net/bundlephobia/minzip/react-resize-detector?style=flat-square"> <img src="https://badgen.net/bundlephobia/tree-shaking/react-resize-detector?style=flat-square">

#### [Live demo](http://maslianok.github.io/react-resize-detector/)

Nowadays browsers support element resize handling natively using [ResizeObservers](https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver). The library uses these observers to help you handle element resizes in React.

🐥 Tiny <a href="https://bundlephobia.com/result?p=react-resize-detector" target="__blank">~3kb</a>

🐼 Written in TypeScript

🦁 Supports Function and Class Components

🐠 Used by <a href="https://github.com/maslianok/react-resize-detector/network/dependents" target="__blank">90k repositories</a>

🦄 Generating <a href="https://npmtrends.com/react-resize-detector" target="__blank">70M+ downloads/year</a>

No `window.resize` listeners! No timeouts! No 👑 viruses! :)

<i>TypeScript-lovers notice: starting from v6.0.0 you may safely remove `@types/react-resize-detector` from you deps list.</i>

## Do you really need this library?

Container queries now work in [all major browsers](https://caniuse.com/css-container-queries). It's very likely you can resolve your problem using [pure CSS](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Container_Queries).

<details><summary>Example</summary>

```html
<div class="post">
  <div class="card">
    <h2>Card title</h2>
    <p>Card content</p>
  </div>
</div>
```

```css
.post {
  container-type: inline-size;
}

/* Default heading styles for the card title */
.card h2 {
  font-size: 1em;
}

/* If the container is larger than 700px */
@container (min-width: 700px) {
  .card h2 {
    font-size: 2em;
  }
}
```

</details>

## Installation

```ssh
npm i react-resize-detector
// OR
yarn add react-resize-detector
```

and

```jsx
import ResizeObserver from 'react-resize-detector';
```

## Examples

Starting from v6.0.0 there are 3 recommended ways to work with `resize-detector` library:

#### 1. React hook (new in v6.0.0)

```jsx
import { useResizeDetector } from 'react-resize-detector';

const CustomComponent = () => {
  const { width, height, ref } = useResizeDetector();
  return <div ref={ref}>{`${width}x${height}`}</div>;
};
```

<details><summary>With props</summary>

```js
import { useResizeDetector } from 'react-resize-detector';

const CustomComponent = () => {
  const onResize = useCallback(() => {
    // on resize logic
  }, []);

  const { width, height, ref } = useResizeDetector({
    handleHeight: false,
    refreshMode: 'debounce',
    refreshRate: 1000,
    onResize
  });

  return <div ref={ref}>{`${width}x${height}`}</div>;
};
```

</details>

<details><summary>With custom ref</summary>

```js
import { useResizeDetector } from 'react-resize-detector';

const CustomComponent = () => {
  const targetRef = useRef();
  const { width, height } = useResizeDetector({ targetRef });
  return <div ref={targetRef}>{`${width}x${height}`}</div>;
};
```

</details>

#### 2. HOC pattern

```jsx
import { withResizeDetector } from 'react-resize-detector';

const CustomComponent = ({ width, height }) => <div>{`${width}x${height}`}</div>;

export default withResizeDetector(CustomComponent);
```

#### 3. Child Function Pattern

```jsx
import ReactResizeDetector from 'react-resize-detector';

// ...

<ReactResizeDetector handleWidth handleHeight>
  {({ width, height }) => <div>{`${width}x${height}`}</div>}
</ReactResizeDetector>;
```

<details><summary>Full example (Class Component)</summary>

```jsx
import React, { Component } from 'react';
import { withResizeDetector } from 'react-resize-detector';

const containerStyles = {
  height: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center'
};

class AdaptiveComponent extends Component {
  state = {
    color: 'red'
  };

  componentDidUpdate(prevProps) {
    const { width } = this.props;

    if (width !== prevProps.width) {
      this.setState({
        color: width > 500 ? 'coral' : 'aqua'
      });
    }
  }

  render() {
    const { width, height } = this.props;
    const { color } = this.state;
    return <div style={{ backgroundColor: color, ...containerStyles }}>{`${width}x${height}`}</div>;
  }
}

const AdaptiveWithDetector = withResizeDetector(AdaptiveComponent);

const App = () => {
  return (
    <div>
      <p>The rectangle changes color based on its width</p>
      <AdaptiveWithDetector />
    </div>
  );
};

export default App;
```

</details>

<details><summary>Full example (Functional Component)</summary>

```jsx
import React, { useState, useEffect } from 'react';
import { withResizeDetector } from 'react-resize-detector';

const containerStyles = {
  height: '100vh',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center'
};

const AdaptiveComponent = ({ width, height }) => {
  const [color, setColor] = useState('red');

  useEffect(() => {
    setColor(width > 500 ? 'coral' : 'aqua');
  }, [width]);

  return <div style={{ backgroundColor: color, ...containerStyles }}>{`${width}x${height}`}</div>;
};

const AdaptiveWithDetector = withResizeDetector(AdaptiveComponent);

const App = () => {
  return (
    <div>
      <p>The rectangle changes color based on its width</p>
      <AdaptiveWithDetector />
    </div>
  );
};

export default App;
```

</details>

<br/>

We still support [other ways](https://github.com/maslianok/react-resize-detector/tree/v4.2.1#examples) to work with this library, but in the future consider using the ones described above. Please let me know if the examples above don't fit your needs.

## Performance optimization

This library uses the native [ResizeObserver](https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver) API.

DOM nodes get attached to `ResizeObserver.observe` every time the component mounts and every time any property gets changed.

It means you should try to avoid passing anonymous functions to `ResizeDetector`, because they will trigger the whole initialization process every time the component rerenders. Use `useCallback` whenever it's possible.

```jsx
// WRONG - anonymous function
const { ref, width, height } = useResizeDetector({
  onResize: () => {
    // on resize logic
  }
});

// CORRECT - `useCallback` approach
const onResize = useCallback(() => {
  // on resize logic
}, []);

const { ref, width, height } = useResizeDetector({ onResize });
```

## Refs

_The below explanation doesn't apply to `useResizeDetector`_

The library is trying to be smart and does not add any extra DOM elements to not break your layouts. That's why we use [`findDOMNode`](https://reactjs.org/docs/reactdom.html#finddomnode) method to find and attach listeners to the existing DOM elements. Unfortunately, this method has been deprecated and throws a warning in StrictMode.

For those who want to avoid this warning, we are introducing an additional property - `targetRef`. You have to set this prop as a `ref` of your target DOM element and the library will use this reference instead of searching the DOM element with help of `findDOMNode`

<details><summary>HOC pattern example</summary>

```jsx
import { withResizeDetector } from 'react-resize-detector';

const CustomComponent = ({ width, height, targetRef }) => <div ref={targetRef}>{`${width}x${height}`}</div>;

export default withResizeDetector(CustomComponent);
```

</details>

<details><summary>Child Function Pattern example</summary>

```jsx
import ReactResizeDetector from 'react-resize-detector';

// ...

<ReactResizeDetector handleWidth handleHeight>
  {({ width, height, targetRef }) => <div ref={targetRef}>{`${width}x${height}`}</div>}
</ReactResizeDetector>;
```

</details>

## API

| Prop            | Type   | Description                                                                                                                                                                                    | Default     |
| --------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| onResize        | Func   | Function that will be invoked with `width` and `height` arguments                                                                                                                              | `undefined` |
| handleWidth     | Bool   | Trigger `onResize` on width change                                                                                                                                                             | `true`      |
| handleHeight    | Bool   | Trigger `onResize` on height change                                                                                                                                                            | `true`      |
| skipOnMount     | Bool   | Do not trigger onResize when a component mounts                                                                                                                                                | `false`     |
| refreshMode     | String | Possible values: `throttle` and `debounce` See [lodash docs](https://lodash.com/docs#debounce) for more information. `undefined` - callback will be fired for every frame                      | `undefined` |
| refreshRate     | Number | Use this in conjunction with `refreshMode`. Important! It's a numeric prop so set it accordingly, e.g. `refreshRate={500}`                                                                     | `1000`      |
| refreshOptions  | Object | Use this in conjunction with `refreshMode`. An object in shape of `{ leading: bool, trailing: bool }`. Please refer to [lodash's docs](https://lodash.com/docs/4.17.11#throttle) for more info | `undefined` |
| observerOptions | Object | These options will be used as a second parameter of [`resizeObserver.observe`](https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver/observe) method.                                | `undefined` |
| targetRef       | Ref    | Use this prop to pass a reference to the element you want to attach resize handlers to. It must be an instance of `React.useRef` or `React.createRef` functions                                | `undefined` |

## Testing with Enzyme and Jest

Thanks to [@Primajin](https://github.com/Primajin) for posting this [snippet](https://github.com/maslianok/react-resize-detector/issues/145)

```jsx
const { ResizeObserver } = window;

beforeEach(() => {
  delete window.ResizeObserver;
  window.ResizeObserver = jest.fn().mockImplementation(() => ({
    observe: jest.fn(),
    unobserve: jest.fn(),
    disconnect: jest.fn()
  }));

  wrapper = mount(<MyComponent />);
});

afterEach(() => {
  window.ResizeObserver = ResizeObserver;
  jest.restoreAllMocks();
});

it('should do my test', () => {
  // [...]
});
```

## License

MIT

## ❤️

Show us some love and STAR ⭐ the project if you find it useful
