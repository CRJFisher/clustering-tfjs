import { defineConfig } from 'vite';

// Project Pages serve under the repo sub-path, not the domain root, so every
// emitted asset URL must carry this prefix or it 404s once deployed. The slug
// is the lowercase GitHub repo name (CRJFisher/clustering-tfjs).
export default defineConfig({
  base: '/clustering-tfjs/',
});
