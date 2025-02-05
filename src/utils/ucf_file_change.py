def replace_base_path(input_file, output_file, old_base_path, new_base_path):
    """
    Replace the base path in a file with updated paths.

    Args:
        input_file (str): Path to the input file (e.g., .txt or .csv).
        output_file (str): Path to save the file with updated paths.
        old_base_path (str): Old base path to be replaced.
        new_base_path (str): New base path to replace the old one.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Check if the line contains a comma (path,label format)
            if ',' in line:
                path, label = line.strip().split(',')
            else:
                path = line.strip()
                label = ""

            # Replace the old base path with the new base path
            if path.startswith(old_base_path):
                updated_path = path.replace(old_base_path, new_base_path, 1)  # Replace only the first occurrence
            else:
                updated_path = path  # If the base path doesn't match, keep it as is

            # Write updated path and label (if exists) to the output file
            if label:
                outfile.write(f"{updated_path},{label}\n")
            else:
                outfile.write(f"{updated_path}\n")


# Example usage
input_file = "C:/Users/user/Desktop/ucf_path_change.txt"  # Replace with your input file
output_file = "C:/Users/user/Desktop/ucf_CLIP_rgbtest_updated.txt"  # Replace with your desired output file
old_base_path = "/home/xbgydx/Desktop/UCFClipFeatures"  # Old base path to replace
new_base_path = "D:/UCFClipFeatures"  # New base path

replace_base_path(input_file, output_file, old_base_path, new_base_path)
