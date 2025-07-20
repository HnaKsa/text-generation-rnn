import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from hazm import Normalizer
import arabic_reshaper
from bidi.algorithm import get_display

# ---------------------------
# 1. بارگذاری و نرمال‌سازی متن فارسی
# ---------------------------

filepath = r"D:\text-generation-rnn\Text generation dataset.txt"
with open(filepath, 'r', encoding='utf-8') as file:
    text = file.read()

normalizer = Normalizer()
text = normalizer.normalize(text)

# ---------------------------
# 2. آماده‌سازی کاراکترها (یا کلمات، ولی اینجا کاراکتر محور است)
# ---------------------------

# کاراکترهای یکتا در متن
characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

seq_length = 40
step_size = 1

sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step_size):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

x = np.zeros((len(sentences), seq_length, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# ---------------------------
# 3. ساخت مدل RNN
# ---------------------------

model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(seq_length, len(characters))))
model.add(LSTM(128))
model.add(Dense(len(characters), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

callbacks = [
    EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

model.fit(x, y, batch_size=128, epochs=50, callbacks=callbacks)
model.save('text_generator_farsi.keras')

# ---------------------------
# 4. تولید متن با دمای مختلف
# ---------------------------

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.maximum(preds, 1e-10)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature=1.0):
    start_index = random.randint(0, len(text) - seq_length - 1)
    sentence = text[start_index: start_index + seq_length]
    generated = sentence

    for i in range(length):
        x_pred = np.zeros((1, seq_length, len(characters)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

# ---------------------------
# 5. چاپ خروجی‌های مختلف
# ---------------------------

def prepare_farsi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)    
    bidi_text = get_display(reshaped_text)           
    return bidi_text

for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f'----------{temp}----------')
    print(prepare_farsi_text(generate_text(300, temperature=temp)))
