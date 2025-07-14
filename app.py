# Importing libraries
import numpy as np
import pickle
from flask import Flask, render_template, request

# Creating an instance of the Flask class
app = Flask(__name__)

# Load model ONCE at the beginning (better performance)
model = pickle.load(open("model.pkl", "rb"))  # Make sure 'model.pkl' is in the same directory

# Index route - serves the form
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')  # Correct path, no need to use flask.render_template

# Prediction logic
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)  # Assuming 12 features
    result = model.predict(to_predict)
    return result[0]

# Route to handle prediction
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # Convert form data to list of integers
            form_data = request.form.to_dict()
            values = list(form_data.values())
            to_predict_list = list(map(int, values))

            # Make prediction
            prediction_result = ValuePredictor(to_predict_list)

            # Interpret result
            if int(prediction_result) == 1:
                prediction = 'Income more than 50K'
            else:
                prediction = 'Income less than 50K'

            return render_template("result.html", prediction=prediction)

        except Exception as e:
            return f"Error: {e}"  # Better error display for debugging

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
