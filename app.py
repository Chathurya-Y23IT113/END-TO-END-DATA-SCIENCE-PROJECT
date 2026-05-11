

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template(
        'index.html',
        prediction_text='',
        area='',
        bedrooms=''
    )

@app.route('/predict', methods=['POST'])
def predict():

    try:
        area = request.form['area']
        bedrooms = request.form['bedrooms']

        features = np.array([[float(area), int(bedrooms)]])

        prediction = model.predict(features)[0]

        result = f'Predicted House Price: Rs.{prediction:,.2f}'

        return render_template(
            'index.html',
            prediction_text=result,
            area=area,
            bedrooms=bedrooms
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}',
            area='',
            bedrooms=''
        )

if __name__ == "__main__":
    app.run(debug=True)

