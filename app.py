from flask import Flask, request, render_template,jsonify
from main import vectorizer,model,preprocess,new_prediction
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  
        review = data.get("review", "")

        preprocessed_text = preprocess(review)
        vectorized_review = vectorizer.transform([preprocessed_text])
        

        prediction = model.predict(vectorized_review)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return jsonify({"sentiment": sentiment})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)