import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";

// A self-contained flat config so CI can `cd site && npm ci && npm run lint`
// without installing the library root. It mirrors the root config's Python-style
// snake_case naming rules — the project's defining convention — and adds the
// browser/worker globals the demo's tfjs workers rely on.
export default tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**", "src/vite-env.d.ts"],
  },
  ...tseslint.configs.recommended,
  eslintConfigPrettier,
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
      globals: {
        window: "readonly",
        document: "readonly",
        navigator: "readonly",
        performance: "readonly",
        self: "readonly",
        postMessage: "readonly",
        Worker: "readonly",
        MessageEvent: "readonly",
        DedicatedWorkerGlobalScope: "readonly",
        HTMLButtonElement: "readonly",
        HTMLPreElement: "readonly",
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "no-empty": "error",
      "prefer-const": "error",
      "@typescript-eslint/no-require-imports": "error",
      "no-constant-condition": "error",
      // Python-style naming: snake_case values, PascalCase types, UPPER_CASE
      // constants. Linear-algebra matrix names (X, A, ...) follow the
      // scikit-learn / math convention and are exempt via the filter regex.
      "@typescript-eslint/naming-convention": [
        "error",
        {
          selector: "default",
          format: ["snake_case"],
          leadingUnderscore: "allow",
          trailingUnderscore: "allow",
        },
        {
          selector: "variable",
          format: ["snake_case", "UPPER_CASE", "PascalCase"],
          leadingUnderscore: "allow",
          trailingUnderscore: "allow",
        },
        {
          selector: "variable",
          filter: { regex: "^_?[A-Z][A-Za-z0-9]*(_[A-Za-z0-9]+)*_?$", match: true },
          format: null,
        },
        {
          selector: "parameter",
          format: ["snake_case"],
          leadingUnderscore: "allow",
          trailingUnderscore: "allow",
        },
        {
          selector: "parameter",
          filter: { regex: "^_?[A-Z][A-Za-z0-9]*(_[A-Za-z0-9]+)*_?$", match: true },
          format: null,
        },
        {
          selector: ["classProperty", "classMethod"],
          format: ["snake_case", "UPPER_CASE"],
          leadingUnderscore: "allow",
          trailingUnderscore: "allow",
        },
        {
          selector: "typeProperty",
          format: ["snake_case"],
          leadingUnderscore: "allow",
          trailingUnderscore: "allow",
        },
        // Single-cap matrix names (X, A, ...) as type properties — kept byte-for-
        // byte aligned with the root eslint.config so the two stay easy to diff.
        {
          selector: "typeProperty",
          filter: { regex: "^[A-Z]$", match: true },
          format: null,
        },
        { selector: "objectLiteralProperty", format: null },
        { selector: "typeLike", format: ["PascalCase"] },
        { selector: "typeParameter", format: ["PascalCase"] },
        { selector: "enumMember", format: ["PascalCase", "UPPER_CASE"] },
        { selector: "import", format: null },
      ],
    },
  },
);
