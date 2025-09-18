import gradio as gr
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib


model = load_model("wine_ann_model.h5")
scaler = joblib.load("scaler.pkl")


skewed_features = ['residual sugar', 'chlorides', 'sulphates', 
                   'free sulfur dioxide', 'total sulfur dioxide', 'fixed acidity']

def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    density, pH, sulphates, alcohol):

    input_df = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

  
    for col in skewed_features:
        input_df[col] = np.log10(input_df[col])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    pred_class = int(np.argmax(prediction, axis=1)[0] + 3)  # shift to original class
    pred_probs = prediction.flatten()
    
    quality_classes = [3, 4, 5, 6, 7, 8]
    proba_dict = {str(c): float(pred_probs[i]) for i, c in enumerate(quality_classes)}
    
    return f"Predicted Quality: {pred_class}", proba_dict


inputs = [
    gr.Number(label="Fixed Acidity", value=7.4),
    gr.Number(label="Volatile Acidity", value=0.7),
    gr.Number(label="Citric Acid", value=0.0),
    gr.Number(label="Residual Sugar", value=1.9),
    gr.Number(label="Chlorides", value=0.076),
    gr.Number(label="Free Sulfur Dioxide", value=11.0),
    gr.Number(label="Total Sulfur Dioxide", value=34.0),
    gr.Number(label="Density", value=0.9978),
    gr.Number(label="pH", value=3.51),
    gr.Number(label="Sulphates", value=0.56),
    gr.Number(label="Alcohol", value=9.4)
]

outputs = [
    gr.Textbox(label="Predicted Quality"),
    gr.Label(num_top_classes=6, label="Prediction Probabilities")
]

gr.Interface(fn=predict_quality, inputs=inputs, outputs=outputs, 
             title="🍷Wine Quality Prediction",
             description="Predict wine quality from chemical features using ANN").launch()
