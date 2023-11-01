React Frontend for HELM
-------------------------

This directory contains the files for building and developing an alternative React based frontend for HELM. If you are looking for the current frontend deployed to https://crfm.stanford.edu/helm/latest/ you will want to look in `helm/benchmark/static` and `helm/benchmark/proxy/static`. If you are looking to make changes to the alternative React frontend, then you are in the correct place.

This app makes use of [React](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/) and built with [vite](https://vitejs.dev/). [Tailwindcss](https://tailwindcss.com/) is used for CSS along with some help from the UI frameworks [daisyUI](https://daisyui.com/) and [tremor](https://www.tremor.so/). [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) is used for tests.



### Installation
```bash
npm Install
```

### Develop

This will open a development server

```bash
npm run dev
```

You will also want to start `helm-server` locally as well. In the `src/helm` directory run the following

```
helm-server
```

### Testing

```
npm run test
```

### Build

```bash
npm run build
```

### Linting

```bash
npm run lint
```

### Formatting

```bash
npm run format
```

### Environment Variables

Requires the following environment variables for development and deployment. In development these can be placed in a `.env.local` file with the following:

```
# The default location of local `helm-server`
VITE_HELM_BENCHMARKS_ENDPOINT="http://localhost:8000/"
# The suites available based on local runs
VITE_HELM_BENCHMARKS_SUITE="v1"
```

This can instead be pointed to the public HELM data to avoid needing to run `helm-server` locally.

```
# Example
VITE_HELM_BENCHMARKS_ENDPOINT="https://storage.googleapis.com/crfm-helm-public/"
# Change to current version
VITE_HELM_BENCHMARKS_SUITE="v0.2.3"
```
# helm-fork
