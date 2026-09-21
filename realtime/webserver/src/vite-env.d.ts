/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional override for the backend origin, e.g. https://box.example:5000 */
  readonly VITE_SERVER_ORIGIN?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
