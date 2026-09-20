import json
import numpy as np
import tensorflow as tf
import nltk
import pickle

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- NLTK ----------------
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# ---------------- CLEAN TEXT ----------------
def clean_text(sentence):
    words = sentence.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# ---------------- LOAD DATA ----------------
with open("dataset_intent_expanded.json", "r", encoding="utf-8") as f:
    data = json.load(f)

patterns = []
tags = []

for intent in data["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        p = clean_text(pattern)  
        patterns.append(p)
        tags.append(tag)

# ---------------- ENCODE TAGS ----------------
unique_tags = sorted(list(set(tags)))
tag_to_index = {t:i for i,t in enumerate(unique_tags)}
y = np.array([tag_to_index[t] for t in tags])

# ---------------- TOKENIZER ----------------
tokenizer = Tokenizer(num_words=8000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)

X = tokenizer.texts_to_sequences(patterns)

max_len = 30
X = pad_sequences(X, maxlen=max_len, padding="post")

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(unique_tags)

# ---------------- SHUFFLE ----------------
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# ---------------- TRANSFORMER BLOCK ----------------
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):

    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)

    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    ffn = Dense(ff_dim, activation="gelu")(x)
    ffn = Dense(inputs.shape[-1])(ffn)

    x = LayerNormalization(epsilon=1e-6)(x + ffn)

    return x

# ---------------- MODEL ----------------
inputs = Input(shape=(max_len,))
x = Embedding(vocab_size, 256)(inputs)
x = transformer_block(x, head_size=64, num_heads=4, ff_dim=256)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="gelu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- CALLBACKS ----------------
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    "chatbot_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ---------------- TRAIN ----------------
print("\nTraining chatbot...\n")

model.fit(
    X,
    y,
    epochs=32,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    shuffle=True
)

# ---------------- SAVE ----------------
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("tags.pkl", "wb") as f:
    pickle.dump(unique_tags, f)

print("Training complete!")