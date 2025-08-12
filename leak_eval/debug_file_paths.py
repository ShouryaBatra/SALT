import os
import sys

def debug_paths():
    """Debug script to find the correct file paths"""
    
    print("=== PATH DEBUGGING ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    print("\n=== LOOKING FOR YOUR FILES ===")
    
    # Try different possible paths for your data file
    possible_paths = [
        "./results/test_pipeline_3sample.json",
        "results/test_pipeline_3sample.json", 
        "../results/test_pipeline_3sample.json",
        "workspace/psbs-research-project/leak_eval/results/test_pipeline_3sample.json",
        "/workspace/psbs-research-project/leak_eval/results/test_pipeline_3sample.json"
    ]
    
    print("Checking possible paths for test_pipeline_3sample.json:")
    for path in possible_paths:
        exists = os.path.exists(path)
        abs_path = os.path.abspath(path)
        print(f"  {path} -> {'✅ EXISTS' if exists else '❌ NOT FOUND'} (absolute: {abs_path})")
    
    print("\n=== DIRECTORY CONTENTS ===")
    
    # Check current directory contents
    print("Current directory contents:")
    try:
        for item in sorted(os.listdir(".")):
            item_path = os.path.join(".", item)
            is_dir = os.path.isdir(item_path)
            print(f"  {'📁' if is_dir else '📄'} {item}")
    except Exception as e:
        print(f"  Error listing current directory: {e}")
    
    # Check if results directory exists
    print("\nLooking for 'results' directory:")
    if os.path.exists("results"):
        print("  📁 results/ directory found!")
        try:
            results_contents = os.listdir("results")
            print("  Contents of results/:")
            for item in sorted(results_contents):
                print(f"    📄 {item}")
        except Exception as e:
            print(f"    Error listing results directory: {e}")
    else:
        print("  ❌ results/ directory not found in current location")
    
    print("\n=== SEARCH FOR FILE ===")
    
    # Search for the file in the filesystem
    print("Searching for test_pipeline_3sample.json...")
    search_dirs = [".", "..", "workspace"] if os.path.exists("workspace") else [".", ".."]
    
    def find_file(directory, filename, max_depth=3, current_depth=0):
        matches = []
        if current_depth >= max_depth:
            return matches
            
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path) and item == filename:
                    matches.append(os.path.abspath(item_path))
                elif os.path.isdir(item_path) and not item.startswith('.'):
                    matches.extend(find_file(item_path, filename, max_depth, current_depth + 1))
        except PermissionError:
            pass
        return matches
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            found_files = find_file(search_dir, "test_pipeline_3sample.json")
            for found_file in found_files:
                print(f"  ✅ FOUND: {found_file}")
    
    print("\n=== RECOMMENDED SOLUTION ===")
    print("Run this debug script first, then update your main script with the correct path.")

if __name__ == "__main__":
    debug_paths()