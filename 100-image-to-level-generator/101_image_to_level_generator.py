import json
import numpy as np
load_images = __import__('110_load_images')
feature_extraction = __import__('120_feature_extraction')
game_levels = __import__('140_turn_features_to_game')



def image_to_level(opts = None):

    imgs_meta = []
    list_of_images = load_images.get_list_of_images_to_load()

    for image_file_meta in list_of_images:
        meta = {}
        meta['file_info'] = image_file_meta
        meta['output_info'] = generate_output_path(image_file_meta)
        meta['raw_img_data'] = load_images.extract_image_data_into_object({"full_filename": image_file_meta["full_filename"]})
        meta['features'] = feature_extraction.extract_features(meta)
        meta['output_levels'] = game_levels.convert_features_to_games(meta)

        imgs_meta.append(meta)

    # with open("./output/output_meta.json", "w") as output_file:
    #     json_dump = json.dumps(imgs_meta, cls=NumpyEncoder)
    #     json.dump(json_dump, output_file, indent=4, sort_keys=True)

def generate_output_path(image_file_meta):
    output_base_path = "./output/"
    output_path = output_base_path + image_file_meta["addition_from_start_path"]

    if (image_file_meta["addition_from_start_path"] != ""):
        output_path += "/"

    output_path += image_file_meta["filename"] + "/"
    return {
        "output_path": output_path
    }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)        

image_to_level()
