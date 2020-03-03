# levels
Turning images into 2-D video game levels

## Installation
```
git clone git@github.com:mattgallivan/levels.git
cd levels
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Example
This will use an example zelda image and the Super Mario Bros. game data along with
a random generator and no repair to generate a 4 x 3 level, printed to standard out.
```
python levels/levels.py zelda.png smb.json random nothing 4 3
```

## Usage Details
```
usage: levels.py [-h] [-og OUTPUT_GEN] [-or OUTPUT_REP] image game generator repair width height

positional arguments:
  image                 Filename of the input RGB image in 'data/imgs/'
  game                  Filename of the game JSON found in 'data/games/'
  generator             The name of the generator function
  repair                The name of the repair function
  width                 Width of the output level
  height                Height of the output level

optional arguments:
  -h, --help            show this help message and exit
  -og OUTPUT_GEN, --output_gen OUTPUT_GEN
                        Filename for generator .txt output
  -or OUTPUT_REP, --output_rep OUTPUT_REP
                        Filename for generator + repair .txt output
```

## How do I write a generator?

Add a function inside generators.py with the following format:

```
def generator_name_here(image, game, width, height):
    ...
```

where image is a PIL image of the input, game is the game's JSON file, and width and height
are the desired width and height of the output level.

## How do I write a repair?

Add a function inside repair.py with the following format:

```
def repair_name_here(level, game, width, height):
    # ...
```

where level is a 2D matrix of the input level and game is the game's JSON file.

## How do I install a library?

```
pip install library_name
pip freeze > requirements.txt
```