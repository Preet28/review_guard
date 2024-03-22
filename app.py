from flask import Flask, render_template, request
import joblib
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the SVM model
model = joblib.load('svm_model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        # Preprocess text function
        def preprocess_text(text):
            stop = set(stopwords.words('english'))
            text = text.lower()
            text = ' '.join([word for word in text.split() if word not in stop])
            return text

        # POS tagging function
        def pos_tagging(text):
            return ' '.join([tag[1] for tag in TextBlob(text).tags])

        processed_review = preprocess_text(review)
        pos_tagged_review = pos_tagging(processed_review)
        
        print("Processed Review:", processed_review)
        print("POS Tagged Review:", pos_tagged_review)
        
        # Transform input text using TF-IDF vectorizer
        review_vectorized = tfidf_vectorizer.transform([pos_tagged_review])
        print("Transformed Review:", review_vectorized)
        
        # Make prediction
        prediction = model.predict(review_vectorized)[0]
        result = "Real Review" if prediction == 1 else "Fake Review"
        print("Prediction:", result)
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
