# 🍷 Wine Quality Prediction 

This project predicts the quality of wine (score between 3 and 8) using its chemical properties.  
It uses an **Artificial Neural Network (ANN)** trained on the **UCI Wine Quality dataset** and provides an **interactive Gradio app** for real-time predictions.

---

## 📂 Project Structure
REDWINE/
│── app.py # Gradio app for wine prediction
│── wine_ann_model.h5 # Trained ANN model
│── scaler.pkl # StandardScaler used for preprocessing
│── README.md # Project documentation

yaml
Copy code

---

## 🚀 Features
- ANN model trained after:
  - Handling duplicates & missing values
  - Correlation analysis
  - Data transformation (log1p on skewed features)
  - Oversampling (SMOTE) to balance classes
  - Feature scaling
- Interactive **Gradio UI** for prediction
- Bar chart visualization of prediction probabilities
- Clean UI with a wine banner image

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wine-quality-ann.git
   cd wine-quality-ann
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Make sure the model and scaler files exist:

wine_ann_model.h5

scaler.pkl

📊 Usage
Run the Gradio app:

bash
Copy code
python app.py
You will see an output like:

nginx
Copy code
Running on local URL:  http://127.0.0.1:7860
Open the link in your browser to access the Wine Quality Prediction App.

🎯 Example Inputs
Feature	Example Value
Fixed Acidity	7.4
Volatile Acidity	0.70
Citric Acid	0.00
Residual Sugar	1.9
Chlorides	0.076
Free Sulfur Dioxide	11.0
Total Sulfur Dioxide	34.0
Density	0.9978
pH	3.51
Sulphates	0.56
Alcohol	9.4

🖼️ App UI
Input wine features

Click Submit

See:

Predicted wine quality

Probability distribution (bar chart)


📌 Requirements
Python 3.8+

TensorFlow / Keras

Scikit-learn

Numpy, Pandas

Matplotlib

Gradio

Install with:

bash
Copy code
pip install tensorflow scikit-learn numpy pandas matplotlib gradio joblib
🏆 Results
Accuracy (Test set): ~0.79

Macro F1-score: ~0.82

Balanced performance across classes after SMOTE oversampling

👨‍💻 Author
Developed by Amr
📌 ANN Model + Gradio UI Integration
