import numpy as np
from flask import Flask,render_template,request
import pickle

app = Flask(__name__,template_folder='template')
sc=pickle.load(open('sc.pkl','rb'))
classifier=pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    Pregnancies=request.form['Pregnancies']
    GlucoseLevel=request.form['Glucose Level']
    BloodPressure=request.form['BloodPressure']
    SkinThickness=request.form['SkinThickness']
    Insulin=request.form['Insulin']
    BMI=request.form['BMI']
    DPF=request.form['DiabetesPedigreeFunction']
    Age=request.form['Age']

    Pregnancies=int(Pregnancies)
    GlucoseLevel=int(GlucoseLevel)
    BloodPressure=int(BloodPressure)
    SkinThickness=int(SkinThickness)
    Insulin=int(Insulin)
    BMI=float(BMI)
    DPF=float(DPF)
    Age=int(Age)
    features=np.array([(Pregnancies,GlucoseLevel,BloodPressure,SkinThickness,Insulin,BMI,DPF,Age)])
    pred=classifier.predict(sc.transform(features))

    if pred==0:
          return render_template('result.html',prediction="Chances of Getting Diabetes is Low")
    if pred==1:
          return render_template('result.html',prediction="Chances of having Diabetes is more")                      

if __name__ == '__main__':
    app.run(debug=True)
