import os.path
from os import path
import shutil

list_subfolders_with_paths = [f.path for f in os.scandir('../html/userContent') if f.is_dir()]

list_subfolders_with_paths
for folder_name in list_subfolders_with_paths:
    has_input_content = os.path.isfile(folder_name + '/input/uploadedImage.png') or os.path.isfile(folder_name + '/input/onlineDrawing.png')
    if not has_input_content:
        print('Removed ' + folder_name)
        shutil.rmtree(folder_name)