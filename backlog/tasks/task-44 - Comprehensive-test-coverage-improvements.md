---
id: task-44
title: Comprehensive test coverage improvements
status: To Do
assignee: []
created_date: '2026-03-20'
labels: []
dependencies: []
---

## Description

The test suite has significant coverage gaps and quality issues identified across all reviewers. Major untested modules: connected_components.ts, component_indicators.ts, spectral_consensus.ts, spectral_optimization.ts, eigen_improved.ts, eigen_qr.ts, som_visualization.ts (6 exports), platform.ts, tf-backend.ts, datasets/synthetic.ts. Test quality issues: Math.random() without seed in 6+ test files causing flaky tests, incorrect afterEach scope cleanup in all 3 validation test files (startScope/endScope is a no-op), areLabelingsEquivalent missing reverse-mapping check (undermines 13 reference tests), only 2 of 12 SOM fixtures tested, very loose tolerance thresholds (50% for QE, 0.3 for U-matrix correlation).

## Acceptance Criteria

- [ ] Unit tests added for connected_components and component_indicators
- [ ] Tests added for spectral_consensus and spectral_optimization
- [ ] Tests added for eigen_improved and eigen_qr including degenerate eigenvalue cases
- [ ] Tests added for som_visualization functions
- [ ] All test data generation uses seeded RNG not Math.random()
- [ ] Validation test afterEach uses correct beforeEach-startScope afterEach-endScope pattern
- [ ] areLabelingsEquivalent checks both forward and reverse mapping
- [ ] SOM reference tests exercise all 12 fixtures not just first 2
- [ ] KMeans tests added for K=1 and K=nSamples and duplicate points and tensor input
- [ ] Integration test added that verifies findOptimalClusters return shape matches declared TypeScript types (from code-charter integration issue #1)
