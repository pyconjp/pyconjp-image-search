import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const useDuckDB = env.VITE_DATASOURCE === "duckdb";

  return {
    plugins: [react()],
    optimizeDeps: {
      exclude: ["@duckdb/duckdb-wasm"],
    },
    build: {
      target: "es2022",
    },
    server: {
      headers: useDuckDB
        ? {
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "credentialless",
          }
        : {},
    },
  };
});
