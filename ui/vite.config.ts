import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

// Development proxy: the Python server (launch_vue) owns /api and /ws on
// port 8000; the Vite dev server forwards so the browser sees one origin
// and no CORS handling is needed.
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000" },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
