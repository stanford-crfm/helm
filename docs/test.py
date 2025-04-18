import os
import glob

def delete_files_with_suffix():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Find all files ending with " 2" recursively
    files_to_delete = []
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if " 2" in file:
                files_to_delete.append(os.path.join(root, file))
    
    # Delete each file
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

if __name__ == "__main__":
    delete_files_with_suffix()
