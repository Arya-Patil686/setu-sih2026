import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site from a subpath, so the base has to match the
// repository name or every asset URL resolves against the domain root and 404s.
// BASE_PATH lets the same build target a root domain or a local preview unchanged.
const base = process.env.BASE_PATH ?? '/setu-sih2026/'

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
  server: {
    port: 5173,
    proxy: {
      // In development the API runs separately; in production `setu serve` mounts the
      // built site behind the same origin, so no proxy is involved.
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
