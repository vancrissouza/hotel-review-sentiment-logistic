#  Hotel Review Sentiment Classifier

This project is a sentiment analysis tool designed to classify hotel reviews as either **positive** or **negative**. It uses a Logistic Regression model trained with TF-IDF (1–3 grams) and class balancing with SMOTE. The app is deployed using **Streamlit**.

---

##  Overview

This classifier helps analyse customer feedback by instantly predicting the sentiment of a hotel review, providing both the label and confidence score. It can be especially useful for hotels looking to gain insights into customer sentiment and improve their service offerings.

---

##  Features

- Text pre-processing with negation handling (`not recommend`)
- TF-IDF vectorisation with unigrams, bigrams and trigrams
- Balanced dataset using SMOTE
- Model trained with Logistic Regression (`class_weight=balanced`)
- Threshold tuning set to **0.6**
- Interactive web app using Streamlit

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vancrissouza/hotel-review-sentiment-logistic.git
   cd hotel-review-sentiment-logistic

2. Set up a virtual environment (optional, but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   .\venv\Scripts\activate  # On Windows


3. Install the dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the app:
   ```bash
   streamlit run app.py
   
---

## Included Notebook

You’ll find the full training pipeline in the notebook:

hotel_sentiment_model_logistic.ipynb

It includes:
- Data cleaning and EDA
- Feature engineering (negation handling)
- TF-IDF + SMOTE pipeline
- Model training and evaluation
- Threshold tuning and result interpretation

It also contains useful visualisations, such as confusion matrices and precision/recall metrics, to help interpret the model's performance.

---

## Example Output

Input:
"I wouldn't recommend this hotel to anyone."

Prediction:
- Sentiment: Negative
- Confidence: 92%

To use the app, simply enter or paste a review in the input field, and the model will return the predicted sentiment along with the associated confidence score.

---

## Repository Structure

hotel-review-sentiment-logistic

├── app.py                      # Main Streamlit app code
├── model.pkl                   # Saved Logistic Regression model
├── vectorizer.pkl              # Saved TF-IDF vectorizer
├── requirements.txt            # List of project dependencies
├── hotel_sentiment_model_logistic.ipynb  # Notebook with training pipeline
└── README.md                   # Project documentation

---

## Created by Vania Souza

Data Analytics & Data Science  
[LinkedIn](https://www.linkedin.com/in/vaniasouzaa/) | [GitHub](https://github.com/vancrissouza)
