module.exports = {
  apps: [
    {
      name: 'captionise-validator',
      script: 'scripts/run_validator.sh', // Use the wrapper script
      autorestart: true,
      watch: false,
    },
  ],
};
