import os

base_directory = "."  # Set this to the path of your base directory if different

# Build a dictionary where keys are filenames and values are lists of paths to those files
files_dict = {}

for dirpath, _, filenames in os.walk(base_directory):
    for filename in filenames:
        if filename not in files_dict:
            files_dict[filename] = []
        files_dict[filename].append(os.path.join(dirpath, filename))

# Create merged files
for filename, paths in files_dict.items():
    # Skip if there's only one file with that name (nothing to merge)
    if len(paths) == 1:
        continue

    merged_file_path = os.path.join(base_directory, f"merged_{filename}")
    with open(merged_file_path, 'w') as merged_file:
        for path in paths:
            with open(path, 'r') as source_file:
                try:
                    merged_file.write(source_file.read())
                except Exception:
                    print(path)

print("Merging completed!")

