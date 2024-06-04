from flask import Flask, request, jsonify, abort
import joblib
import random
import json
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load the model and other necessary files
model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load the intents
with open('intents.json') as file:
    intents = json.load(file)

app = Flask(__name__)
CORS(app)

# Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/message', methods=['POST'])
@limiter.limit("10 per minute") 
def message():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    if len(user_input) > 1000:
        return jsonify({"error": "Message too long"}), 400

    X_test = vectorizer.transform([user_input])
    y_pred = model.predict(X_test)
    intent = label_encoder.inverse_transform(y_pred)[0]

    response = "I'm not sure how to respond to that."
    for i in intents['intents']:
        if i['tag'] == intent:
            response = random.choice(i['responses'])
            break

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
