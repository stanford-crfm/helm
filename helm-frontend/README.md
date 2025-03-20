## React Frontend for HELM

This directory contains the files for building and developing an alternative React based frontend for HELM. If you are looking for the current frontend deployed to https://crfm.stanford.edu/helm/classic/latest/ you will want to look in `helm/benchmark/static` and `helm/benchmark/proxy/static`. If you are looking to make changes to the alternative React frontend, then you are in the correct place.

This app makes use of [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/) and built with [vite](https://vitejs.dev/). [Tailwindcss](https://tailwindcss.com/) is used for CSS along with some help from the UI frameworks [daisyUI](https://daisyui.com/) and [tremor](https://www.tremor.so/). [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) is used for tests.

### Installation

```bash
yarn install
```

### Develop

This will open a development server

```bash
yarn dev
```

You will also want to start `helm-server` locally as well. In the `src/helm` directory run the following

```
helm-server
```

### Testing

```
yarn test
```

### Build

```bash
yarn build
```

### Deployment

For deployment, you can use the build directory out of the box (using the previous command), but will have to replace every leading / in href and src in order for deploys to GitHub Pages to work correctly.

You can rename the build directory to the desired release name and upload it to GitHub pages to add a new front-end release. Currently the GitHub workflow for deploying the React front-end does not do this and is not being used.

### Linting

```bash
yarn lint
```

### Formatting

If you don't have prettier configured in your IDE or Node environment, you will have to run the following before commiting, in order to pass tests.

```bash
yarn format
```
