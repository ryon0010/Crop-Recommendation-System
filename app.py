"""
Crop Recommendation System Web Framework
"""

# Import libraries
from flask import Flask,request,render_template
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier 


model = pickle.load(open('RandomForest.pkl','rb'))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def get_prediction():

    N = request.form['Nitrogen']
    P = request.form['Phosphorus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(data)


    result_str = prediction[0].capitalize()
    result = "{}".format(result_str)
    return render_template('index.html',result = result)
    
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')