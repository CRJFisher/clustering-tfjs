import { defineConfig } from "vite";

// The library's backend loader reaches Node- and React-Native-only TensorFlow.js
// packages through dynamic import()s guarded by runtime platform checks. Those
// branches are dead in the browser, but Rollup still tries to resolve the import
// targets when it bundles the worker — so they are externalized here. The chunks
// that reference them are never fetched in a browser realm, so leaving these as
// bare imports is safe.
const node_only_tf = [
  "@tensorflow/tfjs-node",
  "@tensorflow/tfjs-node-gpu",
  "@tensorflow/tfjs",
  "@tensorflow/tfjs-react-native",
];
// (tfjs-node-gpu is referenced by the library's loader.node alongside tfjs-node.)

// Project Pages serve under the repo sub-path, not the domain root, so every
// emitted asset URL must carry this prefix or it 404s once deployed. The slug
// is the lowercase GitHub repo name (CRJFisher/clustering-tfjs).
export default defineConfig({
  base: "/clustering-tfjs/",
  resolve: {
    // The worker imports @tensorflow/tfjs-core directly AND through
    // clustering-tfjs; both must resolve to ONE physical copy or there would be
    // two tfjs engines — the input tensor would upload to a different engine
    // than the one Clustering.init configured.
    dedupe: ["@tensorflow/tfjs-core"],
  },
  build: {
    rollupOptions: { external: node_only_tf },
  },
  worker: {
    format: "es",
    rollupOptions: { external: node_only_tf },
  },
});
