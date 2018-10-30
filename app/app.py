#!/usr/bin/env python3
import logging

import connexion
from flask_cors import CORS
from flask import render_template
from werkzeug.utils import secure_filename

import prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import torch

def index():
    return render_template('index.html')

def predict(audio_data):
    logger.info(audio_data)
    filename = secure_filename(audio_data.filename)
    path = f'./app/uploads/{filename}.wav'
    audio_data.save(path)

    name = prediction.get_prediction(path)
    return {'name': name}



app = connexion.App(__name__, specification_dir='swagger/',
                              debug=True,
                              swagger_ui=False)
app.add_api('api.yaml')

# add CORS support
CORS(app.app)

app.app.add_url_rule('/', 'index', index)

# set the WSGI application callable to allow using uWSGI:
# uwsgi --http :8080 -w app
application = app.app

if __name__ == '__main__':
    # run our standalone server
    app.run(port=8080)
