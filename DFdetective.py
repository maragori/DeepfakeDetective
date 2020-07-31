# regular stuff
import torch
import inspect
import os
import sys

# flask stuff
from flask import Flask, render_template, request, redirect, url_for

# include parent dir into path for function import
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from DFdetective_utils import (
    init_detective_model,
    predict_from_url
)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# path to the model used for deepfake detection
path_to_model = 'model/'

# cuda stuff
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load model
model = init_detective_model(path_to_model, device)


# by default render the "home.html"
@app.route('/')
def home():
    return render_template('home.html')


# user input triggers prediction script
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        url = request.form['search']

        # if search button hit, call the function predict_from_url
        # this function predicts all clusters and renders a new prediction.html
        predict_from_url(url, model, device)

        # render the new prediction.html
        return redirect(url_for('success', name='Prediction'))


@app.route('/success/<name>')
def success(name):
    return render_template('prediction.html')


if __name__ == '__main__':
    app.config["CACHE_TYPE"] = "null"
    app.run(debug=True) #set threaded=True, processes=4
