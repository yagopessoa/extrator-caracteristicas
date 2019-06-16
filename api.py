from flask import Flask, request
from flask_cors import CORS
import base64

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/get-similars', methods=['POST'])
def get_similars():
    
    code = str(request.data).split('"')[-2]
    imgdata = base64.b64decode(code)
    with open('input.jpg', 'wb') as f:
        f.write(imgdata)

    if request.method == 'POST':

        # Processar e retornar as imagens mais similared

        return 'Success!'