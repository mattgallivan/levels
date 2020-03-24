import json
import numpy as np
load_images = __import__('110_load_images')
feature_extraction = __import__('120_feature_extraction')
game_levels = __import__('140_turn_features_to_game')



def images_to_level(opts = None):

    imgs_meta = []
    list_of_images = load_images.extract_images_file_meta_from_path(opts['start_path'])

    for image_file_meta in list_of_images:
        meta = {}
        meta['file_info'] = image_file_meta
        meta['output_info'] = generate_output_path(image_file_meta, opts['output_path'])
        meta['raw_img_data'] = load_images.extract_image_data_into_object({"full_filename": image_file_meta["full_filename"]})
        meta['features'] = feature_extraction.extract_features(meta, opts)
        meta['output_levels'] = game_levels.convert_features_to_games(meta, opts)

        imgs_meta.append(meta)


    return imgs_meta
    # with open("./output/output_meta.json", "w") as output_file:
    #     json_dump = json.dumps(imgs_meta, cls=NumpyEncoder)
    #     json.dump(json_dump, output_file, indent=4, sort_keys=True)





def generate_output_path(image_file_meta, output_base_path):
    output_path = output_base_path + image_file_meta["addition_from_start_path"]

    if (image_file_meta["addition_from_start_path"] != ""):
        output_path += "/"

    # output_path += image_file_meta["filename"] + "/"
    return {
        "output_path": output_path + image_file_meta["filename"] + "/",
        "output_path_root": output_path
    }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)        




if __name__ == "__main__":

    images_to_level({
        'start_path': '../../data/imgs/drawings', # TEAM: Change this to '../../data/imgs' for more outputs (slower)
        'output_path': '../../output/',
        'games_path': '../../data/games',
        'output': {
            'save_tiles': False
        }
    })
