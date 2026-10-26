import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Keep API requests on the existing Flask backend during local development.
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/process": "http://127.0.0.1:7860",
      "/health": "http://127.0.0.1:7860",
    },
  },
});
