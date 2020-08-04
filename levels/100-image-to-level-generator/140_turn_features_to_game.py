feature_extraction = __import__('120_feature_extraction')
import numpy as np
import cv2
import os
import json
from pathlib import Path

def convert_features_to_games(images_meta, opts):

    games_data = get_compatible_games_info(opts['games_path'])
    save_games_info(images_meta, games_data, {'tile-based matching': 
        {'histogram': list(images_meta['features']['histogram_for_tile_by_tilesize'].keys()), 
         'img': list(images_meta['features']['img_for_tile_by_tilesize'].keys())}
        }, opts )

    images_meta['output'] = {}

    # Matt: I'd add in the trained NN around here.  At this point you have metadata 
    #       information from the image loaded in memory "images_meta['raw_img_data']", 
    #       and if you need it, preprocessed *features* of the images (like tiles of it)
    #       under "images_meta['features']".

    # Conversions utilizing histograms
    if images_meta['features']['histogram_for_tile_by_tilesize'] is not None:
        games_data = turn_gaming_sprites_into_histograms(games_data, opts['games_path'])
        match_assets_into_game_levels(images_meta, games_data, 'histogram')
        save_matched_game_levels(images_meta, games_data, 'histogram', opts)

    # Pixel difference
    if images_meta['features']['img_for_tile_by_tilesize'] is not None:
        match_assets_into_game_levels(images_meta, games_data, 'img')
        save_matched_game_levels(images_meta, games_data, 'img', opts)

    return

def save_games_info(images_meta, games_data, conversion_types, opts = None):

    config = {}
    for games_meta in games_data:
        games_meta['conversions'] = conversion_types
    # games_data['conversions'] = conversion_types
    config['gamesData'] = games_data

    with open(images_meta['output_info']['output_path_root'] + 'config.json', 'w') as outfile:
        json_dump = json.dumps(config, cls=NumpyEncoder)
        # json.dump(json_dump, outfile, indent=4, sort_keys=True)
        outfile.write(json_dump)




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)        


def save_matched_game_levels(images_meta, games_data, asset_type, opts = None):

    for game_name in images_meta['output']:
        output_path = images_meta['output_info']['output_path'] + "games/" + game_name + '/'

        curr_game = {}
        for game_info in games_data:
            if game_info['game_info']['path-friendly-name'] == game_name:
                curr_game = game_info

        # Create path if not exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for tile_size in images_meta['output'][game_name][asset_type+'_match']:
            rows_of_tiles_images = []
            rows_of_tiles_debug_images = []

            with open(output_path + asset_type+"_match_"+str(tile_size)+"px_"+images_meta['file_info']["filename_wo_extension"]+".txt", "w") as output_file:
                for row_of_tiles in images_meta['output'][game_name][asset_type+'_match'][tile_size]:
                    output_file.write("".join(row_of_tiles) + "\n")
                    rows_of_tiles_images.append(cv2.hconcat(list(map(lambda tChar: curr_game['features']['sprites_img'][curr_game['tiles'][tChar]['sprites'][0]], row_of_tiles))))
                    # if opts['output']['save_tiles'] is True:
                    #     border_width = 2
                    #     row_of_tiles_debug_images = list(map(lambda tChar: cv2.copyMakeBorder(curr_game['features']['sprites_img'][curr_game['tiles'][tChar]['sprites'][0]], border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT, None, [255,255,255]), row_of_tiles))
                    #     rows_of_tiles_debug_images.append(cv2.hconcat(row_of_tiles_debug_images))
                cv2.imwrite( output_path + asset_type+"_match_"+str(tile_size)+"px_"+images_meta['file_info']["filename_wo_extension"]+".png", cv2.vconcat(rows_of_tiles_images))
                # if opts['output']['save_tiles'] is True:
                #     cv2.imwrite( output_path + asset_type+"_match_"+str(tile_size)+"px_"+images_meta['file_info']["filename_wo_extension"]+"_debug_show_tilespacing.png", cv2.vconcat(rows_of_tiles_debug_images))



def generate_images_from_ascii_files(opts):
    games_data = get_compatible_games_info(opts['games_path'])
    games_data = turn_gaming_sprites_into_histograms(games_data, opts['games_path'])

    levels_meta = extract_levels_file_meta_from_path(opts['levels_path'])

    for game_data in games_data:
        if game_data['game_info']['path-friendly-name'] == 'super-mario-bros-simplified' or game_data['game_info']['path-friendly-name'] == 'super-mario-bros-sketch-avg':
            for level_meta in levels_meta:
                with open(level_meta['full_filename'], "r") as input_file:
                    rows_of_tiles_images = []
                    for line in input_file:
                        rows_of_tiles_images.append(cv2.hconcat(list(map(lambda tChar: game_data['features']['sprites_img'][game_data['tiles'][tChar]['sprites'][0]], line.replace('\n', '')))))
                    cv2.imwrite(level_meta['path'] + level_meta['filename_wo_extension'] + '-' + game_data['game_info']['path-friendly-name'][16:] +".png", cv2.vconcat(rows_of_tiles_images))

def extract_levels_file_meta_from_path(start_path):

    list_of_levels = []

    # Get list of images
    for path,dirs,files in os.walk(start_path):
        path = path.replace('\\', '/')
        for filename in files:
            filename_wo_extension, extension = os.path.splitext(filename)
            full_filename = os.path.join(path,filename).replace('\\', '/')
            addition_from_start_path = path[len(start_path):]
            if (addition_from_start_path != '' and addition_from_start_path[0] == '/'):
                addition_from_start_path = addition_from_start_path[1:]
            if extension == '.txt':
                list_of_levels.append({
                    "full_filename": full_filename,
                    "filename": filename,
                    "path": path,
                    "filename_wo_extension": filename_wo_extension,
                    "extension": extension,
                    "start_path": start_path, 
                    "addition_from_start_path": addition_from_start_path
                })

    return list_of_levels
                

def match_assets_into_game_levels(images_meta, games_data, asset_type):

    for game_data in games_data:
        if not game_data['game_info']['path-friendly-name'] in images_meta['output']:
            images_meta['output'][game_data['game_info']['path-friendly-name']] = {}

        image_output_meta = images_meta['output'][game_data['game_info']['path-friendly-name']]
        image_output_meta[asset_type + '_match'] = {}

        for tile_size in images_meta['features'][asset_type+'_for_tile_by_tilesize']:
            image_output_meta[asset_type+'_match'][tile_size] = []
            for row_of_tiles in images_meta['features'][asset_type+'_for_tile_by_tilesize'][tile_size]:
                row = []
                for tile_repr in row_of_tiles:
                    row.append(get_best_tile_ASCII(tile_repr, 
                        game_data['features']['sprites_' + asset_type],
                        game_data['sprite_info'], asset_type))
                image_output_meta[asset_type+'_match'][tile_size].append(row)

    return



# def match_histograms_into_game_levels(images_meta, games_data):

#     for game_data in games_data:
#         if not game_data['game_info']['path-friendly-name'] in images_meta['output']:
#             images_meta['output'][game_data['game_info']['path-friendly-name']] = {}

#         image_output_meta = images_meta['output'][game_data['game_info']['path-friendly-name']]
#         image_output_meta['histogram_match'] = {}

#         for tile_size in images_meta['features']['histogram_for_tile_by_tilesize']:
#             image_output_meta['histogram_match'][tile_size] = []
#             for row_of_tiles in images_meta['features']['histogram_for_tile_by_tilesize'][tile_size]:
#                 row = []
#                 for tile_hist in row_of_tiles:
#                     row.append(get_best_tile_ASCII(tile_hist, 
#                         game_data['features']['sprites_histogram'],
#                         game_data['sprite_info']))
#                 image_output_meta['histogram_match'][tile_size].append(row)

#     return


def get_best_tile_ASCII(tile_repr, sprite_repr, sprite_info, asset_type):
    best_sprite_name = compare_game_tiles(tile_repr, sprite_repr, asset_type)
    return sprite_info[best_sprite_name]['char']


def compare_game_tiles(tile_repr, sprite_repr, asset_type):
    results = {}

    for sprite_name in sprite_repr:
        if asset_type == 'histogram':
            results[sprite_name] = cv2.compareHist(tile_repr, sprite_repr[sprite_name], cv2.HISTCMP_BHATTACHARYYA) #HISTCMP_CORREL
        if asset_type == 'img':
            rmax = np.max([tile_repr.shape[0],sprite_repr[sprite_name].shape[0]])
            cmax = np.max([tile_repr.shape[1],sprite_repr[sprite_name].shape[1]])
            max_shape = (rmax,cmax,3)
            tile_img = tile_repr
            sprite_img = sprite_repr[sprite_name]
            
            if tile_repr.shape != max_shape:
                tile_img = np.resize(tile_repr, max_shape)
            if sprite_img.shape != max_shape:
                sprite_img =  np.resize(sprite_img, max_shape)

            results[sprite_name] = np.sum(np.absolute(np.array(tile_img) - np.array(sprite_img)))


    results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
    return results[0][1]

def turn_gaming_sprites_into_histograms(games_data, start_path):

    for game_data in games_data:
        game_data['sprite_info'] = {}
        game_data['features']['sprites_histogram'] = {}
        game_data['features']['sprites_img'] = {}
        path_before_sprites = game_data['file_info']['path'] + '/sprites/'
        for tiles_char in game_data["tiles"]:
            for sprite_name in game_data["tiles"][tiles_char]["sprites"]:
                game_data['sprite_info'][sprite_name] = {
                    "char": tiles_char,
                    "is": game_data["tiles"][tiles_char]["props"]
                }
                game_data['features']['sprites_img'][sprite_name] = \
                    cv2.imread(path_before_sprites 
                                + sprite_name + '.'
                                + game_data['tiles_info']['sprites_extension'])
                game_data['features']['sprites_histogram'][sprite_name] = \
                    feature_extraction.get_histogram_of_image(
                        # cv2.cvtColor(
                            game_data['features']['sprites_img'][sprite_name] 
                            # , cv2.COLOR_BGR2RGB)
                    )

    return games_data

def get_compatible_games_info(start_path):
    
    list_of_games = []

    # Get list of games
    for path,dirs,files in os.walk(start_path):
        path = path.replace('\\', '/')
        for filename in files:
            if (filename != "gameMetadata.json"):
                continue
            filename_wo_extension, extension = os.path.splitext(filename)
            full_filename = os.path.join(path,filename).replace('\\', '/')
            addition_from_start_path = path[len(start_path):]
            if (addition_from_start_path != '' and addition_from_start_path[0] == '/'):
                addition_from_start_path = addition_from_start_path[1:]

            with open(full_filename, 'r') as metadata_file:
                metadata = json.load(metadata_file)
            
            metadata['file_info'] = {
                    "full_filename": full_filename,
                    "filename": filename,
                    "path": path,
                    "filename_wo_extension": filename_wo_extension,
                    "extension": extension,
                    "start_path": start_path, 
                    "addition_from_start_path": addition_from_start_path
            }
            metadata['features'] = {}
            list_of_games.append(metadata)


    return list_of_games



if __name__ == "__main__":
    generate_images_from_ascii_files({
        'levels_path': './generate_images_from_ascii_files/',
        'games_path': '../../data/games',
    })