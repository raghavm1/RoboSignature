#!/bin/bash

# Define the source and destination directories
source_dir="train2014"
destination_dir="train2014500"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

# Find and copy the first 500 files from the source directory
find "$source_dir" -type f | head -n 500 | while read -r file; do
    cp "$file" "$destination_dir"
done
