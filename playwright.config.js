module.exports = {
  use: {
    // Use headless mode
    headless: true,
    // Browser launch options for CI
    launchOptions: {
      args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
    }
  },
  // Only use Chromium for tests
  projects: [
    {
      name: 'chromium',
      use: { 
        browserName: 'chromium'
      }
    }
  ]
};