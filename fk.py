from flask import Flask,request,render_template
from src.mlproject.pipeline.predict_pipeline import Customdata,Pipeline_Predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=Customdata(
            gender=request.form.get(''),
            race_ethnicity=request.form.get(''),
            parental_level_of_education=request.form.get(''),
            lunch=request.form.get(''),
            test_preparation_course=request.form.get(''),
            reading_score=float(request.form.get('')),
            writing_score=float(request.form.get(''))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=Pipeline_Predict()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        