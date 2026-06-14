import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite 开发环境把 /api 转发给 FastAPI，前端代码不需要写死后端地址。
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});

