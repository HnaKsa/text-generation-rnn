import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense , Activation
from tensorflow.keras.optimizers import RMSprop 

filepath = r"D:\text-generation-rnn\Text generation dataset.txt"
text = open(filepath , 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(set(text))

char_to_index = dict((c , i) for i , c in enumerate(characters))
index_to_char = dict((i , c) for i , c in enumerate(characters))

seq_length = 40
step_size = 3

sentences = []
next_characters = []

for i in range(0 , len(text) - seq_length , step_size) :
    sentences.append(text[i: i+seq_length])
    next_characters.append(text[i+seq_length])
    
x = np.zeros((len(sentences), seq_length , len(characters)), dtype= bool)
y = np.zeros((len(sentences), len(characters)), dtype= bool)

for i , sentence in enumerate(sentences) :
    for t , character in enumerate(sentence) :
        x[i , t, char_to_index[character]] = 1 
    y[i , char_to_index[next_characters[i]]] = 1
    
model = Sequential()
model.add(LSTM(128 , input_shape = (seq_length , len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))   

model.compile(loss = 'categorical_crossentropy' , optimizer = RMSprop(learning_rate = 0.01))

model.fit(x , y , batch_size = 256 , epochs = 10)

model.save('text_generator.keras')

# model = tf.keras.models.load_model('text_generator.keras')

def sample(preds , temperature= 1.0) :
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1 , preds , 1)
    return np.argmax(probas)

def generate_text(length , temperature) :
    global characters 
    start_index = random.randint(0 , len(text) - seq_length -1 )
    generated = ''
    sentence = text[start_index : start_index + seq_length]
    generated += sentence
    for i in range(length) : 
        x = np.zeros((1 , seq_length , len(characters)))
        for t , character in enumerate(sentence) :
            x[0 , t , char_to_index[character]] = 1
            
        predictions = model.predict(x , verbose= 0 )[0]
        next_index = sample(predictions , temperature)
        next_character = index_to_char[next_index]
        
        generated += next_character
        sentence = sentence[1:] + next_character
    return generated 

print('----------0.2----------')
print(generate_text(300 , 0.2))
print('----------0.4----------')
print(generate_text(300 , 0.4))
print('----------0.6----------')
print(generate_text(300 , 0.6))
print('----------0.8----------')
print(generate_text(300 , 0.8))
print('----------1----------')
print(generate_text(300 , 1.0))
