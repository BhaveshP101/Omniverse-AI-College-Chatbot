# 🎓 College Chatbot System

An AI-powered college assistant chatbot that answers student queries related to courses, admissions, fees, departments, contact information, and other college-related information.

The system uses a Transformer-based neural network for intent classification and provides a web-based chat interface using Flask.

## 🚀 Features

- AI-powered college information chatbot
- Intent classification using a Transformer-based neural network
- Multi-Head Attention mechanism
- Natural language preprocessing
- Lemmatization using NLTK
- Confidence-based fallback responses
- Flask REST API
- Interactive web chat interface
- Support for college-related FAQs
- Trained Keras model
- JSON-based intent and response dataset

## 🛠️ Technologies Used

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning / NLP
- TensorFlow
- Keras
- NLTK
- NumPy

### Frontend
- HTML
- CSS
- JavaScript

### Data & Model
- JSON
- Pickle
- Keras `.keras` model

## 🧠 Model Architecture

The chatbot uses a Transformer-based neural network consisting of:

1. Tokenization
2. Word Embedding
3. Multi-Head Attention
4. Layer Normalization
5. Feed Forward Network
6. Global Average Pooling
7. Dense Layers
8. Softmax Intent Classification

The predicted intent is mapped to the corresponding response stored in the JSON dataset.

## 📂 Project Structure

```text
college-chatbot/
│
├── app.py
├── chatbot.py
├── training.py
├── index.html
├── dataset_intent_expanded.json
├── chatbot_model.keras
├── tokenizer.pkl
├── tags.pkl
├── requirements.txt
├── .gitignore
└── README.md