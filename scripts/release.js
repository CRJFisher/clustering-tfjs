#!/usr/bin/env node

/**
 * Release automation script
 * Handles version incrementing, tag creation, and PR creation with release label
 * 
 * Can be run from any branch - no need to be on main
 * 
 * Usage:
 *   npm run release patch "Bug fixes and performance improvements"
 *   npm run release minor "New features and enhancements"
 *   npm run release major "Breaking changes"
 *   npm run release patch --file RELEASE_NOTES.md
 *   
 *   # Or specify exact version
 *   npm run release v0.1.1 "Initial release"
 *   npm run release 0.2.0 --file ./docs/release-notes.md
 * 
 * Examples:
 *   npm run release patch "Fixed Windows compatibility and ESLint errors"
 *   npm run release minor "Added clustering algorithms and validation metrics"
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);

// Default to patch if only description is provided
let versionArg = 'patch';
let descriptionStart = 0;

if (args.length === 0) {
  console.error('Usage: npm run release [patch|minor|major] <description>');
  console.error('       npm run release <description>  (defaults to patch)');
  console.error('       npm run release [patch|minor|major] --file <path>');
  console.error('       npm run release <version> <description>');
  console.error('\nExamples:');
  console.error('  npm run release "Fix bug in kmeans"  # patch release');
  console.error('  npm run release minor "Add new feature"');
  console.error('  npm run release v1.2.3 "Custom version"');
  process.exit(1);
}

// Check if first arg is version type or version number
if (args[0] && (
  ['patch', 'minor', 'major'].includes(args[0]) || 
  args[0].match(/^v?\d+\.\d+\.\d+/)
)) {
  versionArg = args[0];
  descriptionStart = 1;
}

const isFile = args[descriptionStart] === '--file';
const descriptionOrPath = isFile 
  ? args[descriptionStart + 1] 
  : args.slice(descriptionStart).join(' ');

if (!descriptionOrPath) {
  console.error('Error: Release description is required');
  process.exit(1);
}

// These will be set after version calculation
let version;
let tagName;

// Get release description
let releaseNotes = '';
if (isFile) {
  if (!descriptionOrPath) {
    console.error('Error: No file path provided after --file');
    process.exit(1);
  }
  try {
    releaseNotes = fs.readFileSync(descriptionOrPath, 'utf8');
  } catch (err) {
    console.error(`Error reading file ${descriptionOrPath}:`, err.message);
    process.exit(1);
  }
} else {
  releaseNotes = descriptionOrPath;
}

// Helper function to run commands
function run(command, options = {}) {
  try {
    return execSync(command, { encoding: 'utf8', stdio: 'pipe', ...options });
  } catch (err) {
    if (!options.ignoreError) {
      console.error(`Command failed: ${command}`);
      console.error(err.message);
      process.exit(1);
    }
    return null;
  }
}

// Helper function to check if we have GitHub CLI
function checkGitHubCLI() {
  try {
    run('gh --version', { ignoreError: true });
    return true;
  } catch {
    return false;
  }
}

// Main release process
async function release() {
  console.log(`🚀 Starting release process...`);

  // Check prerequisites
  if (!checkGitHubCLI()) {
    console.error('Error: GitHub CLI (gh) is required but not installed.');
    console.error('Install it from: https://cli.github.com/');
    process.exit(1);
  }

  // Get current branch
  const currentBranch = run('git branch --show-current').trim();
  console.log(`📍 Creating release from branch: ${currentBranch}`);

  // Ensure working directory is clean
  const gitStatus = run('git status --porcelain').trim();
  if (gitStatus) {
    console.error('Error: Working directory has uncommitted changes.');
    console.error('Please commit or stash your changes first.');
    process.exit(1);
  }

  // Read current version from package.json
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const currentVersion = packageJson.version;
  
  // Calculate new version
  if (['patch', 'minor', 'major'].includes(versionArg)) {
    console.log(`📦 Incrementing ${versionArg} version from ${currentVersion}...`);
    const [major, minor, patch] = currentVersion.split('.').map(Number);
    
    if (versionArg === 'patch') {
      version = `${major}.${minor}.${patch + 1}`;
    } else if (versionArg === 'minor') {
      version = `${major}.${minor + 1}.0`;
    } else if (versionArg === 'major') {
      version = `${major + 1}.0.0`;
    }
    
    tagName = `v${version}`;
    console.log(`   New version: ${version}`);
  } else {
    // Use specific version
    if (!versionArg.match(/^v?\d+\.\d+\.\d+/)) {
      console.error('Error: Invalid version format. Use v0.1.0 or 0.1.0');
      process.exit(1);
    }
    version = versionArg.replace(/^v/, '');
    tagName = `v${version}`;
    console.log(`📦 Setting version to ${version}...`);
  }

  // Check if tag already exists
  const existingTag = run(`git tag -l ${tagName}`, { ignoreError: true });
  if (existingTag && existingTag.trim() === tagName) {
    console.error(`Error: Tag ${tagName} already exists`);
    process.exit(1);
  }
  
  // Update package.json directly
  packageJson.version = version;
  fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2) + '\n');
  
  // Also update package-lock.json if it exists
  if (fs.existsSync('package-lock.json')) {
    const packageLock = JSON.parse(fs.readFileSync('package-lock.json', 'utf8'));
    packageLock.version = version;
    if (packageLock.packages && packageLock.packages['']) {
      packageLock.packages[''].version = version;
    }
    fs.writeFileSync('package-lock.json', JSON.stringify(packageLock, null, 2) + '\n');
  }

  // Commit the version change
  console.log('💾 Committing version change...');
  run('git add package.json package-lock.json');
  run(`git commit -m "chore: release ${version}"`);

  // Create and push tag
  console.log('🏷️  Creating tag...');
  run(`git tag ${tagName}`);

  // Push the branch and tag
  console.log('📤 Pushing branch and tag...');
  run(`git push origin ${currentBranch}`);
  run(`git push origin ${tagName}`);

  // Create PR with release label
  console.log('🔀 Creating pull request...');
  
  // Write release notes to a temporary file for PR body
  const tempFile = path.join(require('os').tmpdir(), `release-notes-${Date.now()}.md`);
  const prBody = `## Release ${tagName}

${releaseNotes}

---
This PR was created automatically by the release script. The tag ${tagName} has already been created and pushed.

Once merged, the release workflow will:
1. Publish to npm
2. Create GitHub release with tag ${tagName}`;
  
  fs.writeFileSync(tempFile, prBody);
  
  try {
    // Create PR with the release label
    const prUrl = run(
      `gh pr create --title "chore: release ${version}" --body-file "${tempFile}" --label "release" --base main`
    ).trim();
    
    console.log(`✨ Pull request created: ${prUrl}`);
    console.log('\n📋 Next steps:');
    console.log('   1. Review the PR');
    console.log('   2. Merge when ready');
    console.log('   3. The release workflow will automatically:');
    console.log('      - Publish to npm');
    console.log('      - Create GitHub release');
    
    // Open PR in browser
    console.log('\n🌐 Opening PR in browser...');
    try {
      run(`gh pr view --web`);
    } catch (err) {
      // Ignore if browser fails to open
    }
  } catch (err) {
    console.error('Failed to create PR:', err.message);
    console.error('You may need to create the PR manually');
  } finally {
    // Clean up temp file
    try {
      fs.unlinkSync(tempFile);
    } catch (err) {
      // Ignore cleanup errors
    }
  }

  console.log('\n🎉 Release preparation complete!');
  console.log(`   Version: ${tagName}`);
  console.log(`   Branch: ${currentBranch}`);
}

// Run the release
release().catch(err => {
  console.error('Unexpected error:', err);
  process.exit(1);
});