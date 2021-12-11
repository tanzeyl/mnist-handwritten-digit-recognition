import os
import pandas as pd
import numpy as np
import flask
import pickle
import joblib
from flask import Flask, render_template, request
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def home():
 return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      file = open("model.pkl","rb")
      trained_model = joblib.load(file)
      with open("Elon Musk.jpg", "r") as f:
         result = trained_model.predict(f)
         return str(result)

if __name__ == "__main__":
 app.run(debug=True)
