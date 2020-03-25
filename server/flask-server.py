from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS
import os.path
from os import path
from random import choice
from string import ascii_lowercase
import hashlib 
import json
from PIL import Image
from werkzeug.utils import secure_filename
import sys

sys.path.insert(0, '../levels/100-image-to-level-generator/')

image_to_level = __import__('101_image_to_level_generator')
  
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__,
            static_url_path='', 
            static_folder='../html')
CORS(app)

base_output_folder = '../dist/userContent'
# base_output_folder = '../html/userContent'

@app.route('/api/generateNewCode', methods=['POST'])
def generateNewCode():
    code = 'love'
    requestedCode = request.json

    if requestedCode['code'] is not '': 
        code = requestedCode['code']

    # Only get a new random code if it's not specifically 
    # requested
    if requestedCode['code'] is '': 
        while path.exists(base_output_folder +  '/' + code):
            code = getCode(5)

    input_folder = base_output_folder +  '/' + code + '/input/temp'
    output_folder = base_output_folder +  '/' + code + '/output'
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metaData = {
        "code": code
    }

    return metaData

def getCode(numChars):
    return ''.join(choice(ascii_lowercase) for i in range(numChars))


@app.route('/api/obtainConfigForCode', methods=['POST'])
def obtainConfigForCode():
    codeMeta = request.json
    with open(base_output_folder + '/' + codeMeta['code'] + '/output' + '/config.json') as json_file:
        config = json.load(json_file)      
        return config  

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
@app.route('/api/uploadImage', methods=['POST'])
def uploadImage():
    formMeta = request.form.to_dict()
    upload_folder = base_output_folder + '/' + formMeta['code'] + '/input/'
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_folder + 'temp/', filename))
            convert_image_to_png(formMeta, upload_folder, filename)
            # Remove temp file that was uploaded
            os.remove(upload_folder + 'temp/' + filename)
            return formMeta
            # redirect(url_for('uploaded_file',
            #                         filename=filename))
    return 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_image_to_png(formMeta, upload_folder, filename):
    im = Image.open(upload_folder + 'temp/' + filename)
    im.save(upload_folder + formMeta['image_type'] + '.png')


@app.route('/api/turnImageIntoGameContent', methods=['POST'])
def turnImageIntoGameContent():
    formMeta = request.json
    print(formMeta)
    upload_folder = base_output_folder +  '/' + formMeta['code']
    if request.method == 'POST':
        images_meta = image_to_level.images_to_level({
            'start_path': upload_folder + '/input/',
            'output_path': upload_folder + '/output/',
            'games_path': '../data/games',
            'output': {
                'save_tiles': False
            }
        })

        with open(upload_folder + '/output/' + 'config.json') as json_file:
            config = json.load(json_file)      
            return config  
        
        return formMeta


@app.route('/api/checkCode')
def checkCode():
    print('Check code')




    # with open(base_output_folder +  '/' + code + '/config.json', 'w') as outfile:
    #     json.dump(config, outfile)

    # # print(code)
    # # print(os.getcwd())
    # # print(path.exists(base_output_folder +  '/djbd'))
    # # print(path.exists(base_output_folder +  '/love'))





if __name__ == "__main__":
    app.run(host='0.0.0.0')
