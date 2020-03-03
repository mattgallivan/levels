import random as r


def random(image, game, width, height):
    tiles = list(game["tiles"].keys())
    return [[r.choice(tiles) for _ in range(height)] for _ in range(width)]
