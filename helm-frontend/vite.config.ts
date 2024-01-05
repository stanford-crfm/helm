import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  base: "",
  plugins: [react()],
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
    outDir: `${__dirname}/../src/helm/benchmark/static_build`,
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom', 'react-markdown', 'react-router-dom', 'react-spinners'],
          tremor: ["@tremor/react"],
          recharts: ["recharts"],
        }
      }
    }
  },
});
