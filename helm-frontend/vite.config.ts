import { defineConfig, ViteDevServer } from "vite";
import serveStatic from "serve-static";
import react from "@vitejs/plugin-react";
import path from "path";

const ServeBenchmarkOutputPlugin = {
  name: 'serve-benchmark-output-plugin',
  configureServer(server: ViteDevServer) {
    server.middlewares.use(
      "/benchmark_output",
      // TODO: Make the path configurable via environment variable
      serveStatic("../benchmark_output", {fallthrough: false, index: false})
    )
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  base: "",
  plugins: [react(), ServeBenchmarkOutputPlugin],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
  },
  build: {
    rollupOptions: {
      output: {
        // Manually chunk large libraries to keep chunk size under 500 KB
        manualChunks: {
          react: ["react", "react-dom", "react-markdown", "react-router-dom", "react-spinners"],
          tremor: ["@tremor/react"],
          recharts: ["recharts"],
          yaml: ["yaml"],
        }
      }
    }
  },
});
