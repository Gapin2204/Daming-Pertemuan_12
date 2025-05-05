from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model
model, labels = pickle.load(open('model_karang.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ambil input dari form
    features = [float(x) for x in request.form.values()]
    features_array = np.array([features])
    pred_idx = model.predict(features_array)[0]
    pred_label = labels[pred_idx]
    
    return render_template('index.html', prediction_text=f'Jenis karang: {pred_label}')

if __name__ == "__main__":
    app.run(debug=True)
