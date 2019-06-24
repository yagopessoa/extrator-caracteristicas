# 
# $env:FLASK_APP = "api.py"
# 
# flask run
# 

from flask import Flask, request
from flask_cors import CORS

import base64
import extract_feature

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return 'API - Extrator de Características'

@app.route('/get-similars', methods=['POST'])
def get_similars():
    if request.method == 'POST':

        code = str(request.data).split('"')[-2]
        imgdata = base64.b64decode(code)
        with open('input.jpg', 'wb') as f:
            f.write(imgdata)

        resp = extract_feature.main()
        return str(resp)
