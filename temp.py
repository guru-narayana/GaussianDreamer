import os
import shutil

def delete_dir_contents(dir_path):
    """Deletes all files and folders within a given directory.

    Args:
        dir_path (str): The path to the directory to clear.
    """

    for filename in os.listdir(dir_path):
        if filename == "save":
            continue
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
           os.remove(file_path)  # Remove files
        elif os.path.isdir(file_path):
           shutil.rmtree(file_path)  # Recursively remove subdirectories
# Example usage:
dir_to_clear = "outputs/multigaussiandreamer-vsd/A_rainbow-colored_umbrella@20240605-104900"
delete_dir_contents(dir_to_clear)

print(f"Contents of '{dir_to_clear}' have been deleted.")