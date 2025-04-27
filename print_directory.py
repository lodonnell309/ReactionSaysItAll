import os

def print_directory_structure(root_dir, indent=0):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        print("  " * indent + "|-- " + item)
        if os.path.isdir(item_path):
            print_directory_structure(item_path, indent + 1)

if __name__ == "__main__":
    root_directory = "."  # Current directory as default
    print(root_directory)
    print_directory_structure(root_directory)