import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)
import joblib
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

# @app.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'),
#                                'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
    name = request.form.get('name')

    if name:
        prediction = predict_loan_sanction([18, 5000, 700, 1])
        print('Request for hello page received with name=%s: ' % name + str(prediction))
        return render_template('hello.html', name=name, prediction=prediction)
    else:
        print('Request for hello page received with no name or blank name -- redirecting')
        return redirect(url_for('index'))



def predict_loan_sanction(input_values):
    loaded_model = joblib.load('model/loan_prediction_model.pkl')
    input_array = np.array(input_values).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return prediction[0]



if __name__ == '__main__':
   app.run()
