# Contributing to clustering-tfjs

Thank you for your interest in contributing to clustering-tfjs! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Project Structure](#project-structure)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Prerequisites

- Node.js 18.x or 20.x
- npm 8.x or higher
- Git

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/clustering-tfjs.git
   cd clustering-tfjs
   ```

3. Install dependencies:

   ```bash
   npm install
   ```

4. Run tests to ensure everything is working:

   ```bash
   npm test
   ```

## Development Workflow

This project uses the Backlog.md system for task management. Please read the guidelines in `CLAUDE.md` for detailed instructions.

### Finding Work

```bash
# List available tasks
backlog task list -s "To Do" --plain

# View task details
backlog task 42 --plain
```

### Starting Work

1. Assign yourself to a task:

   ```bash
   backlog task edit 42 -a @YOUR_USERNAME -s "In Progress"
   ```

2. Create a new branch:

   ```bash
   git checkout -b task-42-description
   ```

3. Add implementation plan to the task before coding

### During Development

- Write tests for new functionality
- Update documentation as needed
- Follow the coding standards below
- Run tests frequently: `npm test`
- Check types: `npm run type-check`
- Run linter: `npm run lint`

## Coding Standards

### TypeScript Guidelines

- Use TypeScript for all new code
- Provide explicit types for function parameters and return values
- Avoid `any` type unless absolutely necessary
- Use interfaces for object shapes
- Document complex types with JSDoc comments

### Code Style

```typescript
// ✅ Good
export function calculateDistance(
  pointA: number[],
  pointB: number[]
): number {
  // Implementation
}

// ❌ Bad
export function calculateDistance(pointA, pointB) {
  // Implementation
}
```

### Tensor Management

When working with TensorFlow.js tensors:

1. **Always dispose tensors** when done:

   ```typescript
   const tensor = tf.tensor2d(data);
   try {
     // Use tensor
   } finally {
     tensor.dispose();
   }
   ```

2. **Use tf.tidy()** for automatic cleanup:

   ```typescript
   const result = tf.tidy(() => {
     const a = tf.tensor2d(data1);
     const b = tf.tensor2d(data2);
     return a.add(b);
   });
   ```

3. **Check tensor disposal** in validation functions - don't dispose user-provided tensors

### Import Organization

1. External dependencies first
2. Internal imports grouped by type
3. Type imports last

```typescript
import * as tf from "@tensorflow/tfjs-node";
import { Matrix } from "ml-matrix";

import { KMeans } from "../clustering/kmeans";
import { pairwiseDistance } from "../utils/distance";

import type { DataMatrix, LabelVector } from "../types";
```

## Testing Guidelines

### Test Structure

- Unit tests: `test/unit/`
- Integration tests: `test/integration/`
- Algorithm tests: `test/clustering/`
- Utility tests: `test/utils/`

### Writing Tests

```typescript
describe("AlgorithmName", () => {
  beforeAll(() => {
    tf.setBackend("tensorflow");
  });

  afterEach(() => {
    // Clean up tensors
    tf.engine().startScope();
    tf.engine().endScope();
  });

  it("should produce expected results", async () => {
    // Arrange
    const data = [[1, 2], [3, 4]];
    
    // Act
    const result = await algorithm.fit(data);
    
    // Assert
    expect(result).toEqual(expected);
  });
});
```

### Scikit-learn Parity

When implementing algorithms, ensure compatibility with scikit-learn:

1. Use the comparison tools in `tools/sklearn_comparison/`
2. Generate test fixtures with Python scripts
3. Test edge cases and parameter variations
4. Document any intentional deviations

## Documentation

### Code Documentation

- Add JSDoc comments to all exported functions
- Include parameter descriptions and examples
- Document complex algorithms with references

```typescript
/**
 * Performs K-means clustering on the input data.
 * 
 * @param X - Input data matrix (n_samples × n_features)
 * @param options - Clustering options
 * @returns Cluster labels for each sample
 * 
 * @example
 * ```typescript
 * const labels = await kmeans.fitPredict([[1, 2], [3, 4]]);
 * ```
 */
```

### Task Documentation

When completing a task, add an "Implementation Notes" section:

```markdown
## Implementation Notes

- Approach taken and why
- Key technical decisions
- Any deviations from the plan
- Files modified or created
```

## Submitting Changes

### Before Submitting

1. Ensure all tests pass: `npm test`
2. Run type checking: `npm run type-check`
3. Run linter: `npm run lint`
4. Update task documentation with implementation notes
5. Mark acceptance criteria as complete

### Pull Request Process

1. Push your branch to your fork
2. Create a pull request against the main repository
3. Use a clear title: "Task XX: Brief description"
4. Include in the PR description:
   - Task number and link
   - Summary of changes
   - Test results
   - Any breaking changes

### PR Review

- Address reviewer feedback promptly
- Keep discussions focused and professional
- Update your branch by rebasing if needed

## Project Structure

```
clustering-tfjs/
├── src/
│   ├── clustering/      # Algorithm implementations
│   ├── utils/           # Helper functions
│   ├── validation/      # Metrics (silhouette, etc.)
│   └── index.ts         # Main exports
├── test/                # Test files
├── tools/               # Development tools
│   └── sklearn_comparison/  # Scikit-learn comparison
├── backlog/             # Task management
│   ├── tasks/          # Active tasks
│   ├── docs/           # Project documentation
│   └── decisions/      # Architecture decisions
└── benchmarks/          # Performance benchmarks
```

## Questions?

If you have questions:

1. Check existing issues and discussions
2. Review the task documentation in `backlog/`
3. Ask in the PR or create an issue

Thank you for contributing to clustering-tfjs!
