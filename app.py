from flask import Flask, request, render_template, flash, url_for
from flask import Response
from werkzeug.utils import redirect
import psycopg2
from AirBnb import compute_predictions
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory
from Airbnb_config import FeatureSelection, DateTimeColumns, Geo ,host_response_rate ,treat_missing_first, categorical_encoder, treat_missing_second ,Scaler_Min_Max, Mydimension_reducer
import pandas as pd
import numpy as np

app=Flask(__name__,template_folder='Html')
UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "AIApps"

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/AllApps')
def AllApps():
    return render_template('All_Apps.html')

@app.route('/AirBnB')
def AirBnB():
    return render_template('AirBnB.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_airbnb', methods = ['GET', 'POST'])
def upload_file():
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            ##########################Compute###########################
            Predictions=compute_predictions('Uploads/'+filename)
            # Predictions=compute_predictions('Uploads/',filename)


            os.remove('Uploads/'+filename)

 
            ##########################Export###########################
            resp = Response(Predictions.to_csv())
            # flash("Task Completed!")
            resp.headers["Content-Disposition"] = "attachment; filename=AirBNB_Predictions.csv"
            resp.headers["Content-Type"] = "text/csv"
            return resp
           
            # 
            # return render_template('AirBnB.html')


if __name__  == '__main__':
    app.run(debug=True)