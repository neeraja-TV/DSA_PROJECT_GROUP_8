from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

knn_model= joblib.load('knn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = [float(request.form['feature1']), float(request.form['feature2']),float(request.form['feature3']),float(request.form['feature4']),
                  float(request.form['feature5']),float(request.form['feature6']),float(request.form['feature7']),float(request.form['feature8']),
                  float(request.form['feature9']),float(request.form['feature10']),float(request.form['feature11']),float(request.form['feature12']),
                  float(request.form['feature13']),float(request.form['feature14']),float(request.form['feature15']),float(request.form['feature16']),
                  float(request.form['feature17']),float(request.form['feature18']),float(request.form['feature19']),float(request.form['feature20']),
                  float(request.form['feature21']),float(request.form['feature22']),float(request.form['feature23']),float(request.form['feature24']),
                  float(request.form['feature25']),float(request.form['feature26']),float(request.form['feature27'])]

    # Preprocess and scale input data
    scaled_data = np.array([input_data])  # Assuming min-max scaling

    # Make predictions using k-NN model
    prediction = knn_model.predict(scaled_data)
    result = knn_model.predict(scaled_data)
    if result == 0:
        prediction = "Nordic walking"
    elif result == 1:
        prediction = "ascending stairs"
    elif result == 2:
        prediction = "cycling"
    elif result == 3:
        prediction = "descending stairs "
    elif result == 4:
        prediction = "ironing "
    elif result == 5:
        prediction = "lying"
    elif result == 6:
        prediction = " rope jumping"
    elif result == 7:
        prediction = "running"
    elif result == 8:
        prediction = "sitting"
    elif result == 9:
        prediction = "standing"
    elif result == 10:
        prediction = "transient activities"
    elif result == 11:
        prediction = "vacuum cleaning"
    elif result == 12:
        prediction = "walking"
    else:
        prediction="Please check range values and enter again"

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=8000)




