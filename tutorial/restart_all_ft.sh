#!/bin/bash

cd "$(dirname "$0")"

base_dir="."

# Array of strategy directories
strategies=(
     Simple_Scalping
     claude
     chatgpt
)

# Function to stop a strategy
remove_strategy() {
    local strategy=$1

    echo "Stopping $strategy..."
    cd "$base_dir/$strategy"
    docker compose down
    cd - > /dev/null  # Return to previous directory silently
    sleep 2
}

# Function to deploy a strategy
deploy_strategy() {
    local strategy=$1

    echo "Starting $strategy..."
    cd "$base_dir/$strategy"
    docker compose build
    docker compose up -d
    cd - > /dev/null  # Return to previous directory silently
    sleep 31
}

for strategy in "${strategies[@]}"; do
    remove_strategy "$strategy"
done

docker system prune -a -f
sleep 31

# Deploy all strategies
for strategy in "${strategies[@]}"; do
    deploy_strategy "$strategy"
done
