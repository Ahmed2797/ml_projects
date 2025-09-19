import os
import sys
import pandas as pd
from flask import Flask, render_template, request
from src.mlproject.pipeline.predict_pipeline import *

app = Flask(__name__)
pipe = Predictpipeline()

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def Home_prediction():
    if request.method == 'GET':
        return render_template('home1.html')
    else:
        try:
            data = Custom(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )

            df = data.getdata_as_dataframe()
            result = pipe.predict(df)

            return render_template('home1.html', result=result[0])
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    app.run(debug=True)
