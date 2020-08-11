# regular stuff
import torch
import inspect
import os
import sys

# flask stuff
from flask import Flask, render_template, request, redirect, url_for, Response

# include parent dir into path for function import
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from DFdetective_utils import (
    init_detective_model,
    predict_from_url
)

app = Flask(__name__)

# path to the model used for deepfake detection
path_to_model = 'model/'

# cuda stuff
device = torch.device("cpu")

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

        path = predict_from_url(url, model, device)

        name = path.split('/')[-1] + '_prediction'

        # render the new prediction.html
        return redirect(url_for('success', name=name))


@app.route('/success/<name>')
def success(name):
    return render_template(name + '.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")
