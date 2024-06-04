import json
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Load the intents
with open('intents.json') as file:
    data = json.load(file)

# Preprocess the data
sentences = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Vectorize the sentences
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer
import joblib
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
