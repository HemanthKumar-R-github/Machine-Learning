from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('regression_model.pkl')  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input
    crime_rate = float(request.form['crime_rate'])
    residential_area = float(request.form['residential_area'])
    age = float(request.form['age'])
    airport = int(request.form['airport'])
    # Make predictions using your trained model
    # Assuming a global variable `model` exists
    
    prediction = model.predict([[crime_rate, residential_area,0.5,6.0, age, 23.0,9.0,airport,0.005]])
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
