import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from hazm import Normalizer, word_tokenize
import arabic_reshaper
from bidi.algorithm import get_display

# === 1. بارگذاری و نرمال‌سازی متن فارسی ===
filepath = r"D:\text-generation-rnn\Text generation dataset.txt"
with open(filepath, encoding='utf-8') as f:
    text = f.read()

normalizer = Normalizer()
text = normalizer.normalize(text)  # نیم‌فاصله، ی / ي ، ک / ك

# === 2. توکنایز کلمه‌ای ===
tokens = word_tokenize(text)

# === 3. ساخت توالی‌های ورودی و خروجی ===
seq_length = 5
input_sequences = []
for i in range(seq_length, len(tokens)):
    seq = tokens[i-seq_length:i+1]
    input_sequences.append(seq)

# === 4. تبدیل به اعداد ===
tokenizer = Tokenizer()
tokenizer.fit_on_texts([' '.join(seq) for seq in input_sequences])
word_index = tokenizer.word_index
index_word = {i: w for w, i in word_index.items()}
total_words = len(word_index) + 1

sequences = []
for seq in input_sequences:
    encoded = tokenizer.texts_to_sequences([' '.join(seq)])[0]
    sequences.append(encoded)

sequences = np.array(sequences)
x, y = sequences[:, :-1], sequences[:, -1]

# === 5. تعریف مدل ===
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=128, input_length=seq_length))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# === 6. آموزش مدل ===
callbacks = [
    EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('farsi_text_generator.keras', save_best_only=True)
]

model.fit(x, y, epochs=50, batch_size=128, callbacks=callbacks)

# === 7. تابع تولید متن فارسی ===
def generate_text(seed_text, next_words=50, temperature=0.7):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')

        preds = model.predict(token_list, verbose=0)[0]
        preds = np.asarray(preds).astype('float64')
        preds = np.maximum(preds, 1e-10)
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        next_index = np.random.choice(range(len(preds)), p=preds)
        next_word = index_word.get(next_index, '')
        seed_text += ' ' + next_word
    return seed_text

# # === 8. تست تولید متن ===
# print(generate_text("روزی روزگاری", next_words=50, temperature=0.7))


def prepare_farsi_text(text):
    reshaped_text = arabic_reshaper.reshape(text)    
    bidi_text = get_display(reshaped_text)           
    return bidi_text

for temp in [0.2, 0.4, 0.6, 0.8, 1.0]:
    print(f'----------{temp}----------')
    print(prepare_farsi_text(generate_text(300, temperature=temp)))
