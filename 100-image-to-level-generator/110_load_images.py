import cv2
import os


def get_list_of_images_to_load(opts = None):

    list_of_images = []

    # Get list of images
    start_path = './data/imgs' # current directory
    for path,dirs,files in os.walk(start_path):
        path = path.replace('\\', '/')
        for filename in files:
            filename_wo_extension, extension = os.path.splitext(filename)
            full_filename = os.path.join(path,filename).replace('\\', '/')
            addition_from_start_path = path[len(start_path):]
            if (addition_from_start_path != '' and addition_from_start_path[0] == '/'):
                addition_from_start_path = addition_from_start_path[1:]
            list_of_images.append({
                "full_filename": full_filename,
                "filename": filename,
                "path": path,
                "filename_wo_extension": filename_wo_extension,
                "extension": extension,
                "start_path": start_path, 
                "addition_from_start_path": addition_from_start_path
            })

    # Filter
    # TBD: filter out only for images we want to process

    # Return list of images
    return list_of_images


def extract_image_data_into_object(opts):
    output = {}
    output['opencv'] = cv2.imread(opts['full_filename']) #cv2.cvtColor(cv2.imread(opts['full_filename']), cv2.COLOR_BGR2RGB) 
    return output

