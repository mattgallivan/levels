def write_level(filename, level):
    with open(filename, 'w') as txt_file:
        txt_file.writelines("".join(str(tile) for tile in row) + "\n" for row in level)
