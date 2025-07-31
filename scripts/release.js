#!/usr/bin/env node

/**
 * Release automation script
 * Handles tag creation, monitors GitHub Actions, and updates release notes
 * 
 * Usage:
 *   node scripts/release.js <version> <description>
 *   node scripts/release.js <version> --file <path-to-release-notes.md>
 *   node scripts/release.js v0.1.0 "Initial release with clustering algorithms"
 *   node scripts/release.js v0.1.1 --file RELEASE_NOTES.md
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length < 2) {
  console.error('Usage: node scripts/release.js <version> <description>');
  console.error('       node scripts/release.js <version> --file <path>');
  process.exit(1);
}

const version = args[0];
const isFile = args[1] === '--file';
const descriptionOrPath = isFile ? args[2] : args.slice(1).join(' ');

if (!version.match(/^v?\d+\.\d+\.\d+/)) {
  console.error('Error: Invalid version format. Use v0.1.0 or 0.1.0');
  process.exit(1);
}

// Ensure version starts with 'v'
const tagName = version.startsWith('v') ? version : `v${version}`;

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
  console.log(`🚀 Starting release process for ${tagName}...`);

  // Check prerequisites
  if (!checkGitHubCLI()) {
    console.error('Error: GitHub CLI (gh) is required but not installed.');
    console.error('Install it from: https://cli.github.com/');
    process.exit(1);
  }

  // Ensure we're on main branch
  const currentBranch = run('git branch --show-current').trim();
  if (currentBranch !== 'main') {
    console.error(`Error: Must be on main branch. Currently on: ${currentBranch}`);
    process.exit(1);
  }

  // Ensure working directory is clean
  const gitStatus = run('git status --porcelain').trim();
  if (gitStatus) {
    console.error('Error: Working directory has uncommitted changes.');
    console.error('Please commit or stash your changes first.');
    process.exit(1);
  }

  // Pull latest changes
  console.log('📥 Pulling latest changes...');
  run('git pull origin main');

  // Check if tag already exists
  const existingTag = run(`git tag -l ${tagName}`, { ignoreError: true });
  if (existingTag && existingTag.trim() === tagName) {
    console.error(`Error: Tag ${tagName} already exists`);
    process.exit(1);
  }

  // Create and push tag
  console.log(`🏷️  Creating tag ${tagName}...`);
  run(`git tag ${tagName}`);
  
  console.log('📤 Pushing tag to GitHub...');
  run(`git push origin ${tagName}`);

  // Get the workflow run ID
  console.log('⏳ Waiting for GitHub Actions workflow to start...');
  
  // Wait a bit for the workflow to be triggered
  await new Promise(resolve => setTimeout(resolve, 5000));

  // Find the workflow run
  let workflowRunId = null;
  let attempts = 0;
  const maxAttempts = 12; // 1 minute total

  while (!workflowRunId && attempts < maxAttempts) {
    try {
      // Get all recent workflow runs
      const runsJson = run(`gh api repos/{owner}/{repo}/actions/runs --jq '.'`, { ignoreError: true });
      if (runsJson) {
        const runsData = JSON.parse(runsJson);
        
        // Find runs triggered by our tag
        const tagRuns = runsData.workflow_runs.filter(run => 
          run.head_branch === tagName || 
          run.head_sha === run(`git rev-list -n 1 ${tagName}`, { ignoreError: true })?.trim()
        );
        
        if (tagRuns.length > 0) {
          // Get the most recent run
          workflowRunId = tagRuns[0].id;
          console.log(`Found workflow run: ${workflowRunId}`);
        }
      }
    } catch (err) {
      // Ignore JSON parse errors
    }

    if (!workflowRunId) {
      attempts++;
      console.log(`   Waiting for workflow to start... (${attempts}/${maxAttempts})`);
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }

  if (!workflowRunId) {
    console.warn('⚠️  Could not find workflow run. You may need to update the release manually.');
    console.log('Creating release anyway...');
  } else {
    // Monitor workflow status
    console.log('📊 Monitoring workflow progress...');
    let status = 'queued';
    let conclusion = null;

    while (status === 'queued' || status === 'in_progress') {
      await new Promise(resolve => setTimeout(resolve, 10000)); // Check every 10 seconds
      
      try {
        const runInfo = JSON.parse(run(`gh api repos/{owner}/{repo}/actions/runs/${workflowRunId}`));
        status = runInfo.status;
        conclusion = runInfo.conclusion;
        
        console.log(`   Status: ${status}${conclusion ? ` (${conclusion})` : ''}`);
      } catch (err) {
        console.error('Error checking workflow status:', err.message);
        break;
      }
    }

    if (conclusion !== 'success') {
      console.error(`❌ Workflow failed with status: ${conclusion}`);
      console.error('The tag has been pushed, but the npm publish may have failed.');
      console.error(`Check the workflow at: https://github.com/{owner}/{repo}/actions/runs/${workflowRunId}`);
      process.exit(1);
    }

    console.log('✅ Workflow completed successfully!');
  }

  // Create or update GitHub release
  console.log('📝 Creating GitHub release...');
  
  // Write release notes to a temporary file to avoid shell escaping issues
  const tempFile = path.join(require('os').tmpdir(), `release-notes-${Date.now()}.md`);
  fs.writeFileSync(tempFile, releaseNotes);
  
  try {
    run(`gh release create ${tagName} --title "${tagName}" --notes-file "${tempFile}"`);
    console.log(`✨ Release ${tagName} created successfully!`);
    console.log(`View at: https://github.com/{owner}/{repo}/releases/tag/${tagName}`);
  } catch (err) {
    // Release might already exist if manually created
    console.log('Release may already exist, attempting to edit...');
    try {
      run(`gh release edit ${tagName} --notes-file "${tempFile}"`);
      console.log('✨ Release notes updated successfully!');
    } catch (err2) {
      console.error('Failed to create or update release:', err2.message);
    }
  } finally {
    // Clean up temp file
    try {
      fs.unlinkSync(tempFile);
    } catch (err) {
      // Ignore cleanup errors
    }
  }

  console.log('\n🎉 Release process complete!');
  console.log(`   Version: ${tagName}`);
  console.log(`   Package: https://www.npmjs.com/package/clustering-js`);
}

// Run the release
release().catch(err => {
  console.error('Unexpected error:', err);
  process.exit(1);
});