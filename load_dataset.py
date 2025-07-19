import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# --- 1. Load text from your .txt file ---
with open(r"D:\text-generation-rnn\Text generation dataset.txt", encoding='utf-8') as f:
    text = f.read()