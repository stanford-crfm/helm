#!/bin/bash

# Define the output file
combined_output="combined_ngrams"

# Empty the output file if it exists
> $combined_output

# Loop through all files with "_ngrams" in the name and append to the combined output
for file in output_stats/*_ngrams; do
    cat "$file" >> $combined_output
done

echo "All ngrams files have been combined into $combined_output"

