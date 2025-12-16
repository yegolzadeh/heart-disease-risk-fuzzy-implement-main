from flask import Flask, request, render_template, flash
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from heart_disease_fuzzy import predict_risk

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Required for flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check allowed file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Process uploaded dataset and return stats + summary table
def process_dataset(df):
    if df.empty:
        return None, "<p>The dataset is empty.</p>"

    # Map inputs according to the paper
    df['chest_pain_mapped'] = df['cp'].map({1: 7, 2: 5, 3: 3, 4: 1})
    df['hba1c'] = np.where(df['fbs'] == 1, 9.0, 5.5)
    df['hdl'] = df['chol'] * 0.25
    df['ldl'] = df['chol'] * 0.45
    df['heart_rate'] = df['thalach']
    df['age_mapped'] = df['age']
    df['bp_mapped'] = df['trestbps']

    # Predict risk for all patients
    risks = []
    levels = []
    for _, row in df.iterrows():
        try:
            risk_val = predict_risk(
                row['chest_pain_mapped'],
                row['hba1c'],
                row['hdl'],
                row['ldl'],
                row['heart_rate'],
                row['age_mapped'],
                row['bp_mapped']
            )
            risks.append(round(risk_val, 2))
            level = "Healthy" if risk_val < 4 else "Low Risk" if risk_val < 6 else "Medium Risk" if risk_val < 8 else "High Risk"
            levels.append(level)
        except:
            risks.append(0.0)
            levels.append("Error")

    df['predicted_risk'] = risks
    df['risk_level'] = levels

    # Summary statistics
    stats = {
        'total': len(df),
        'avg_risk': round(df['predicted_risk'].mean(), 2),
        'high_risk_count': len(df[df['risk_level'] == 'High Risk']),
        'high_risk_percent': round((len(df[df['risk_level'] == 'High Risk']) / len(df)) * 100, 1)
    }

    # Sample table (first 10 + last 10 rows)
    cols = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'predicted_risk', 'risk_level']
    summary_html = pd.concat([df.head(10), df.tail(10)])[cols].to_html(classes='table table-striped', index=False)

    return stats, summary_html

@app.route('/', methods=['GET', 'POST'])
def index():
    manual_result = None
    uploaded_stats = None
    uploaded_table = "<p>No file has been uploaded yet.</p>"

    # Handle manual form submission
    if request.method == 'POST' and 'chest_pain' in request.form:
        try:
            cp = float(request.form['chest_pain'])
            hba = float(request.form['hba1c'])
            hd = float(request.form['hdl'])
            ld = float(request.form['ldl'])
            hr = float(request.form['heart_rate'])
            ag = float(request.form['age'])
            bp = float(request.form['blood_pressure'])

            risk_value = round(predict_risk(cp, hba, hd, ld, hr, ag, bp), 2)
            level = "Healthy" if risk_value < 4 else "Low Risk" if risk_value < 6 else "Medium Risk" if risk_value < 8 else "High Risk"
            manual_result = f"Risk Score: {risk_value} → {level}"
        except Exception as e:
            manual_result = "Error: Please enter valid numeric values."

    # Handle file upload
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                uploaded_df = pd.read_csv(filepath)
                uploaded_stats, uploaded_table = process_dataset(uploaded_df)
                flash(f'File "{filename}" uploaded and analyzed successfully!')
            except Exception as e:
                flash(f'Error processing uploaded file: {str(e)}')
        else:
            flash('Only CSV files are allowed.')

    return render_template('index.html',
                           manual_result=manual_result,
                           uploaded_stats=uploaded_stats,
                           uploaded_table=uploaded_table)

if __name__ == '__main__':
    app.run(debug=True)