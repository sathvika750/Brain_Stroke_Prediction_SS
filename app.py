from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import shap

app = Flask(__name__, template_folder="../frontend")

model_path = os.path.join("model", "model.joblib")
stroke_model = joblib.load(model_path)

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    encoded_cols, numeric_cols = stroke_model["encoded_cols"], stroke_model["numeric_cols"]
    preprocessor = stroke_model["preprocessor"]
    
    input_df[encoded_cols] = preprocessor.transform(input_df)
    X = input_df[numeric_cols + encoded_cols]
    
    # Prediction

    prediction = stroke_model['model'].predict(X)[0]
    

    explainer = shap.LinearExplainer(stroke_model["model"], X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)
    
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "impact": shap_values[0] 
    }).sort_values(by="impact", key=abs, ascending=False).head(3)
    
    explanation = ", ".join([f"{row.feature} ({row.impact:.2f})" for _, row in feature_importance.iterrows()])
    
    precautions = []
    if single_input['hypertension']:
        precautions.append("Monitor your blood pressure regularly and reduce salt intake.")
    if single_input['bmi'] > 25:
        precautions.append("Maintain a healthy weight through diet and exercise.")
    if single_input['smoking_status'] != "never smoked":
        precautions.append("Consider quitting smoking to reduce stroke risk.")
    if single_input['avg_glucose_level'] > 140:
        precautions.append("Maintain healthy blood sugar levels and monitor regularly.")
    if single_input['heart_disease']:
        precautions.append("Consult your cardiologist for regular checkups.")
    
    return prediction, explanation, precautions


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gender = request.form["gender"].lower()
        age = int(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        ever_married = request.form["ever_married"].lower()
        work_type = request.form["work_type"]
        residence_type = request.form["residence_type"]
        avg_glucose_level = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking_status = request.form["smoking_status"].lower()

        work_type_mapping = {
            "Government job": "Govt_job",
            "Children": "children",
            "Never Worked": "Never_worked",
            "Private": "Private",
        }

        single_input = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type_mapping.get(work_type, work_type),
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }


    
        prediction, explanation, precautions = predict_input(single_input)
        result = "Likely" if prediction == 1 else "Not Likely"

        # Only show precautions if stroke is likely
        if prediction == 0:
            precautions = []

        return render_template("result.html", result=result, explanation=explanation, precautions=precautions)


    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
