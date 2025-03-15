#!/bin/bash

# Exit on error
set -e

echo "Setting up Caption Subnet development environment..."

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Set up database
echo "Setting up PostgreSQL database..."
if command -v psql &> /dev/null; then
    # If PostgreSQL is installed, create database
    if psql -lqt | cut -d \| -f 1 | grep -qw caption_subnet; then
        echo "Database 'caption_subnet' already exists"
    else
        createdb caption_subnet || echo "Failed to create database (might already exist)"
    fi
else
    echo "PostgreSQL not found. Please install PostgreSQL or use Docker for development."
fi

# Set up Bittensor wallets if needed
if ! command -v btcli &> /dev/null; then
    echo "Bittensor CLI not found. Please install Bittensor before proceeding."
    exit 1
fi

# Create wallets if they don't exist
echo "Checking Bittensor wallets..."
if [ ! -d "$HOME/.bittensor/wallets/miner" ]; then
    echo "Creating miner wallet..."
    btcli wallet new_coldkey --wallet.name miner --no_password --no_prompt
    btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default --no_prompt
fi

if [ ! -d "$HOME/.bittensor/wallets/validator" ]; then
    echo "Creating validator wallet..."
    btcli wallet new_coldkey --wallet.name validator --no_password --no_prompt
    btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default --no_prompt
fi

echo "Setup complete! You can now run the miner or validator:"
echo "- python neurons/caption_miner.py --netuid 1 --wallet.name miner --wallet.hotkey default"
echo "- python neurons/caption_validator.py --netuid 1 --wallet.name validator --wallet.hotkey default" 