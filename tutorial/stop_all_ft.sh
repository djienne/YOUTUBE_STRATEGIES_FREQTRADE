#!/bin/bash

base_dir="."

# Array of strategy directories
strategies=(
     Simple_Scalping
     claude
     chatgpt
)


remove_strategy() {
    local strategy=$1

    echo "Stopping $strategy..."
    cd "$base_dir/$strategy"
    docker compose down
    cd - > /dev/null  # Return to previous directory silently
}


for strategy in "${strategies[@]}"; do
    remove_strategy "$strategy"
done

docker system prune -a -f
