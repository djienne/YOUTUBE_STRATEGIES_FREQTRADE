#!/bin/bash

base_dir="."

# Array of strategy directories
strategies=(
     Simple_Scalping
     claude
     chatgpt
)

# Function to deploy a strategy
deploy_strategy() {
    local strategy=$1

    echo "Starting $strategy..."
    cd "$base_dir/$strategy"
    docker compose build
    docker compose up -d
    cd - > /dev/null  # Return to previous directory silently
    sleep 61
}

# Deploy all strategies
for strategy in "${strategies[@]}"; do
    deploy_strategy "$strategy"
done
