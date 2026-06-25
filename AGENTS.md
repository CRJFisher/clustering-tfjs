# Practices

## Code conventions

- **Naming is Python-style:** `snake_case` for variables, functions, methods, properties, parameters, and file names; `PascalCase` for classes, interfaces, type aliases, and enums; `UPPER_SNAKE_CASE` for constants. No `camelCase` anywhere.
- **Source is organized by domain/action**, not generic buckets. See the source-layout list in `AGENTS.md`. There is no `utils/` folder.
- **Tests are colocated** next to the source they cover (`foo.ts` ↔ `foo.test.ts`); shared reference fixtures live in `__fixtures__/`.

## Utilities for debugging Spectral Clustering implementation against sklearn

- in `tools/sklearn_fixtures` there is a `.venv` with sklearn installed. This can be used to run debug scripts to check intermediate results against sklearn.
- in `sklearn_reference` there is the actual sklearn implementation of spectral clustering. It can be used to check the implementation against sklearn. If any further code is needed, please download it and add it to the `sklearn_reference` directory.
