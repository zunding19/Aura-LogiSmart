import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Python server (python server.py) answers /api on port 8000.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
