module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint", "import"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:import/errors",
    "plugin:import/warnings",
    "plugin:import/typescript",
    "prettier",
  ],
  env: {
    node: true,
    jest: true,
    es2020: true,
  },
  settings: {
    "import/resolver": {
      typescript: {},
    },
  },
};

