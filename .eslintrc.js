module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  plugins: ["@typescript-eslint"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "prettier",
  ],
  env: {
    node: true,
    jest: true,
    es2020: true,
  },
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: "module",
    project: "./tsconfig.json",
  },
  rules: {
    // Allow any types in specific cases where needed
    "@typescript-eslint/no-explicit-any": "warn",
    // Allow unused vars with underscore prefix
    "@typescript-eslint/no-unused-vars": [
      "warn",
      { "argsIgnorePattern": "^_", "varsIgnorePattern": "^_" }
    ],
    // Allow empty blocks
    "no-empty": "warn",
    // Allow const reassignment in some cases
    "prefer-const": "warn",
    // Allow require statements
    "@typescript-eslint/no-var-requires": "warn",
    // Disable constant condition check for now
    "no-constant-condition": "warn",
  },
};

