import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function loadHttps() {
  try {
    return {
      key: fs.readFileSync(resolve(__dirname, 'server.key')),
      cert: fs.readFileSync(resolve(__dirname, 'server.cert')),
    };
  } catch {
    return undefined;
  }
}
const httpsOptions = loadHttps();

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        viewer: resolve(__dirname, 'viewer.html'),
        sender: resolve(__dirname, 'sender.html'),
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    // Self-signed certs are gitignored and only exist on a dev box. Load them
    // if present (browsers refuse getUserMedia over plain http), but don't make
    // their absence fatal -- `vite build` needs no TLS, and behind a tunnel the
    // tunnel terminates TLS for us.
    ...(httpsOptions ? { https: httpsOptions } : {}),
  },
  css: {
    postcss: './postcss.config.js',
  },
});
