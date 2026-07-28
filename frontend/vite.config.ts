import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API runs separately and is never bundled. Proxying keeps the browser
    // on one origin so CORS config is not load-bearing for local development.
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
