# The code is importing three modules: `os`, `imageio`, and `re`.
import os
import imageio
import re


def sort_strings_with_suffix(strings):
    """
    The function `sort_strings_with_suffix` sorts a list of strings based on a numerical suffix
    extracted from each string using regular expressions.
    
    Args:
      strings: The parameter "strings" is a list of strings that we want to sort. Each string may or may
    not have a numeric suffix at the end.
    
    Returns:
      The function `sort_strings_with_suffix` returns a sorted list of strings, where the sorting is
    based on the numerical suffix present in each string.
    """
    def extract_suffix(string):
        # Extract the suffix using regular expression
        suffix = re.findall(r'\d+', string)
        
        if suffix:
            return int(suffix[0])
        else:
            return string

    return sorted(strings, key=lambda s: extract_suffix(s))


def generate_gif(path, file_prefix, output_path, output_file, duration):
    """
    The function `generate_gif` takes in a path, file prefix, output path, output file name, and
    duration, and generates a GIF by reading images from the specified path, sorting them, and saving
    them with the specified duration.
    
    Args:
      path: The path parameter is the directory where the input images are located.
      file_prefix: The file prefix is a string that is used to filter the files in the given path. Only
    the files that have the file prefix in their names will be included in the GIF generation process.
      output_path: The output path is the directory where the generated GIF will be saved.
      output_file: The output_file parameter is the name of the GIF file that will be generated.
      duration: The duration parameter specifies the time duration (in seconds) for each frame in the
    generated GIF.
    """
   
    output_path = "images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    
    frames = []
    list_files = os.listdir(path)
    list_files_sorted = sort_strings_with_suffix(list_files)
    for file in list_files_sorted:
        if file_prefix in file:
            print(file)
            image = imageio.v2.imread(os.path.join(path,file)) 
            frames.append(image)
    imageio.mimsave(os.path.join(output_path, output_file),
                frames,          
                duration = duration)