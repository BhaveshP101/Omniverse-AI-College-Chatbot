import json
import random
import numpy as np
import tensorflow as tf
import nltk
import pickle

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import load_model

# ---------------- NLTK ----------------
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# ---------------- CLEAN TEXT ----------------

from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


def clean_text(sentence):
    sentence = sentence.lower()
    #automatic spelling correction
    #sentence = str(TextBlob(sentence).correct())

    words = sentence.split()
    return " ".join(words)

# LOAD MODEL 
model = load_model("chatbot_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("tags.pkl", "rb") as f:
    unique_tags = pickle.load(f)

# ---------------- LOAD RESPONSES ----------------
with open("dataset_intent_expanded.json", "r", encoding="utf-8") as f:
    data = json.load(f)

responses = {}
for intent in data["intents"]:
    responses[intent["tag"]] = intent["responses"]

index_to_tag = {i:t for i,t in enumerate(unique_tags)}

max_len = 30

fallback_responses = [
    "Sorry, I cannot answer that.",
    "Please ask a proper question.",
    "I don't understand.",
    "Can you rephrase your question?"
]

# CHAT FUNCTION 
def chatbot_reply(text):

    text = clean_text(text)   

    seq = tokenizer.texts_to_sequences([text])[0]

    if len(seq) == 0:
        return random.choice(fallback_responses)

    seq_pad = pad_sequences([seq], maxlen=max_len, padding="post")

    pred = model.predict(seq_pad, verbose=0)[0]

    confidence = np.max(pred)
    index = np.argmax(pred)

    if confidence < 0.60:   
        return random.choice(fallback_responses)

    tag = index_to_tag[index]

    return random.choice(responses[tag])


# ---------------- CHAT LOOP ----------------
if __name__ == "__main__":
    print("\nChatbot Ready! (type 'exit')\n")

    while True:
        user = input("You: ")

        if user.lower() == "exit":
            break

        print("Bot:", chatbot_reply(user))