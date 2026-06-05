import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier";

export default tseslint.config(
  {
    ignores: [
      "dist/**",
      "node_modules/**",
      "coverage/**",
      "__fixtures__/**",
      "**/*.d.ts",
    ],
  },
  ...tseslint.configs.recommended,
  eslintConfigPrettier,
  {
    files: [
      "src/**/*.ts",
      "test_support/**/*.ts",
      "benchmarks/**/*.ts",
    ],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: "module",
      parserOptions: {
        project: "./tsconfig.eslint.json",
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
      // Python-style naming: snake_case values, PascalCase types, UPPER_CASE constants.
      // Linear-algebra matrix names (X, A, U_full, X_train, ...) follow the
      // scikit-learn / math convention and are exempt via `matrixName`.
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
        {
          selector: "typeProperty",
          filter: { regex: "^[A-Z]$", match: true },
          format: null,
        },
        // Object-literal keys are often external/data contracts — don't enforce.
        { selector: "objectLiteralProperty", format: null },
        { selector: "typeLike", format: ["PascalCase"] },
        { selector: "typeParameter", format: ["PascalCase"] },
        { selector: "enumMember", format: ["PascalCase", "UPPER_CASE"] },
        // Imported bindings keep their source name (e.g. TF.js API).
        { selector: "import", format: null },
      ],
    },
  },
);
