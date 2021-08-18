from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import joblib
import pickle
import os
nltk.download('punkt')

app = Flask(__name__)

model = joblib.load('joblib_model.pkl')
tfidf_vect = pickle.load(open('tfidf.pickle', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

# this function to classify the excel file with the ML model that we made before
@app.route("/classify", methods=['GET', 'POST'])
def add_model_in_excel():
    #os.chdir("path/to")
    #print os.path.abspath(excel_file)
    DataInAList = []
    new = [] 

    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        file = request.form['upload']
        df = pd.read_excel(file)
        corpus = df['Designation'].values
        for sent in corpus:
            new = tfidf_vect.transform([str(sent)])
            prediction = model.predict(new)
            res_prediction = '?'
            if (prediction == 0):
                res_prediction = "MPR"
            if (prediction == 1):
                res_prediction = "BAT"
            if (prediction == 2):
                res_prediction = "INFO"
            if (prediction == 3):
                res_prediction = "MOB"
        
            DataInAList.append(res_prediction)
        df["Prediction sous cat"] = DataInAList
        newExcel = df.to_excel("NewExcel.xlsx",index=False)
    return render_template('index.html', newExcel = newExcel)

# the last function to download the new excel file
@app.route("/download")
def download_excel():
    path = "NewExcel.xlsx"
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')