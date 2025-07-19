import re
import numpy as np
from load_dataset import text

# --- 2. Clean the text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^آ-یءa-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_text = clean_text(text)
tokens = cleaned_text.split()
