from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pretrained model and label encoders
model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        person_id = request.form['person_id']
        person_age = request.form['person_age']
        person_income = request.form['person_income']
        person_home_ownership = request.form['person_home_ownership']
        person_emp_length = request.form['person_emp_length']
        loan_intent = request.form['loan_intent']
        loan_grade = request.form['loan_grade']
        loan_amnt = request.form['loan_amnt']
        loan_int_rate = request.form['loan_int_rate']
        loan_percent_income = request.form['loan_percent_income']
        cb_person_default_on_file = request.form['cb_person_default_on_file']
        cb_person_cred_hist_length = request.form['cb_person_cred_hist_length']

        # Check if any required field is empty
        if not person_age or not person_income or not person_emp_length or not loan_amnt or not loan_int_rate or not person_id:
            return render_template('error.html', message="Please fill in all required fields.")

        # Convert to float after checking the value is not empty
        person_id = float(person_id)
        person_age = float(person_age)
        person_income = float(person_income)
        person_home_ownership = label_encoders['person_home_ownership'].transform([person_home_ownership])[0]  # Apply LabelEncoder
        person_emp_length = float(person_emp_length)
        loan_intent = label_encoders['loan_intent'].transform([loan_intent])[0]  # Apply LabelEncoder
        loan_grade = label_encoders['loan_grade'].transform([loan_grade])[0]  # Apply LabelEncoder
        loan_amnt = float(loan_amnt)
        loan_int_rate = float(loan_int_rate)
        loan_percent_income = float(loan_percent_income)
        cb_person_default_on_file = label_encoders['cb_person_default_on_file'].transform([cb_person_default_on_file])[0]  # Apply LabelEncoder
        cb_person_cred_hist_length = float(cb_person_cred_hist_length)

        # Prepare the input data for prediction (ensure the array matches the expected features)
        input_data = np.array([[person_id, person_age, person_income, person_home_ownership, person_emp_length,
                                loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
                                cb_person_default_on_file, cb_person_cred_hist_length]])

        # Make prediction directly with the model
        prediction = model.predict(input_data)

        # Display the result
        result = "Approved" if prediction == 1 else "Denied"
        return render_template('result.html', prediction=result)

    except Exception as e:
        # Catch any errors and display a generic message
        return render_template('error.html', message=str(e))


if __name__ == '__main__':
    app.run(debug=True)
