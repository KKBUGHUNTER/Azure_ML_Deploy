import os
import joblib
import numpy as np 

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')


@app.route('/hello', methods=['POST'])
def hello():
	a1 = request.form.get('a1')
	a2 = request.form.get('a2')
	a3 = request.form.get('a3')
	a4 = request.form.get('a4')
	
	input_values = [int(a1), int(a2), int(a3), int(a4)]
	predicted_amount = predict_loan_sanction(input_values)
	print("Predicted Loan Sanction Amount:", predicted_amount)
	
	if predicted_amount:
		return render_template('hello.html', name = str(predicted_amount))
	else:
		return redirect(url_for('index'))


def predict_loan_sanction(input_values):
	loaded_model = joblib.load('model/loan_prediction_model.pkl')
	input_array = np.array(input_values).reshape(1, -1)
	prediction = loaded_model.predict(input_array)
	return prediction[0]


if __name__ == '__main__':
   app.run()
