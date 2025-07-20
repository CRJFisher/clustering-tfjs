// Jest is executed in a read-only file-system sandbox in the autograder.
// Some dependencies (e.g. TensorFlow.js) attempt to write temporary files
// which would throw errors. We monkey-patch the fs module to make all write
// operations no-ops. This file is loaded via the `-r` CLI flag in the `npm
// test` script.

const fs = require("fs");

const noOp = () => {};

const writeSyncMethods = [
  "writeFileSync",
  "appendFileSync",
  "mkdirSync",
  "rmSync",
  "rmdirSync",
  "unlinkSync",
  "copyFileSync",
  "renameSync",
];

const writeAsyncMethods = [
  "writeFile",
  "appendFile",
  "mkdir",
  "rm",
  "rmdir",
  "unlink",
  "copyFile",
  "rename",
];

for (const m of writeSyncMethods) {
  if (typeof fs[m] === "function") {
    fs[m] = noOp;
  }
}

for (const m of writeAsyncMethods) {
  if (typeof fs[m] === "function") {
    fs[m] = (...args) => {
      const cb = args.pop();
      if (typeof cb === "function") cb(null);
    };
  }
}

module.exports = {};

/* ------------------------------------------------------------------------- */
/*               Patch third-party helpers that bypass our stubs              */
/* ------------------------------------------------------------------------- */

try {
  // `write-file-atomic` is used by Jest to persist the transform cache. We
  // monkey-patch its export to turn all write attempts into no-ops so that Jest
  // falls back to keeping the transformed code in memory.
  const wfa = require("write-file-atomic");
  wfa.sync = noOp;
  wfa.writeFileSync = noOp;
  wfa.async = (...args) => {
    const cb = args.pop();
    if (typeof cb === "function") cb(null);
  };
} catch {
  /* ignore – dependency may not be present */
}
