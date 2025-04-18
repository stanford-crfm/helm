import os
import glob

def delete_files_with_suffix():
    # Get the current directory
    current_dir = os.getcwd()
    
    # Find all files ending with " 2"
    files_to_delete = glob.glob(os.path.join(current_dir, "* 2*"))
    
    # Delete each file
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

if __name__ == "__main__":
    delete_files_with_suffix()
