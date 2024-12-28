from flask import Flask, request, render_template
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import load

app = Flask(__name__)

# Load the trained model
model = None

def load_model():
    global model
    if model is None:
        # Load the model from disk
        model = load('trained_model.joblib')





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    if model:
        # Get the input data from the form
        data = {
            'department': request.form['department'],
            'salary_range': float(request.form['salary_range']),
            'telecommuting': int(request.form['telecommuting']),
            'has_company_logo': int(request.form['has_company_logo']),
            'has_questions': int(request.form['has_questions']),
            'employment_type': request.form['employment_type'],
            'required_experience': request.form['required_experience'],
            'required_education': request.form['required_education'],
            'industry': request.form['industry'],
            'function': request.form['function']
        }
        # Make a prediction
        prediction = model.predict(pd.DataFrame([data]))[0]
        return render_template('result.html', prediction=prediction)
    else:
        return 'Model not loaded'

if __name__ == '__main__':
    app.run(debug=True)
