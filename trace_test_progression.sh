#!/bin/bash

echo "=== Tracing test progression through commits ==="
echo

# First, stash current changes
echo "Stashing current changes..."
git stash push -m "temp stash for test progression check"

# Check test results at key commits
commits=(
  "HEAD"  # Current state
  "81b4967"  # Before eigenvector recovery
  "1d6295d"  # After k-NN fix
)

for commit in "${commits[@]}"; do
  echo "=== Checking commit $commit ==="
  git checkout $commit 2>/dev/null || { echo "Failed to checkout $commit"; continue; }
  
  # Build
  npm run build > /dev/null 2>&1
  
  # Run test
  echo "Running fixture tests..."
  npx ts-node verify_fixture_results.ts 2>/dev/null | grep -E "(ARI =|Passed:|Failed:)" | tail -3
  echo
done

# Return to original state
echo "Returning to original state..."
git checkout -
git stash pop

echo "Done!"