#!/bin/bash

# Find all directories named "logs" in the current directory tree
log_folders=$(find . -type d -name "logs")

# Check if any logs folders were found
if [ -z "$log_folders" ]; then
    echo "No logs folders found."
    exit 1
fi

# Initialize a variable to track files to delete
files_to_delete=""

# Loop through each logs folder and find files to delete
for folder in $log_folders; do
    # Find freqtrade.log.* files in the current logs folder
    folder_files=$(find "$folder" -type f -name "freqtrade.log.*")

    # Append found files to files_to_delete if any exist
    if [ -n "$folder_files" ]; then
        files_to_delete+="$folder_files"$'\n'
    fi
done

# Check if there are files to delete
if [ -z "$files_to_delete" ]; then
    echo "No rotated log files found to delete."
    exit 0
fi

# Display files and ask for confirmation
echo "The following files will be deleted:"
echo "$files_to_delete"
echo -n "Do you want to proceed? (y/n): "
read confirmation

if [[ "$confirmation" =~ ^[Yy]$ ]]; then
    # Delete the files
    echo "$files_to_delete" | xargs rm -f
    echo "Rotated log files have been deleted."
else
    echo "Operation cancelled. No files were deleted."
fi
