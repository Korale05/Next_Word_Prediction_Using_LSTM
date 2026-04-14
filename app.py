import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Load the Lstm model and tokinizar
model = load_model(r'Next_Word_Prediction\next_word_lstm.keras')


with open(r"Next_Word_Prediction\tokenizer.pickle","rb") as file:
    tokenizer = pickle.load(file)


# Function to predict the next word
def sample_with_temperature(preds):
    return np.random.choice(len(preds), p=preds)

def predict_next_word(model,tokenizer,text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list])

    predicted = model.predict(token_list,verbose=0)[0]

    predicted_word_index = sample_with_temperature(predicted)

    return tokenizer.index_word.get(predicted_word_index)


## Streamlate app
st.title("Next word Prediction with LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of word",'To be or not to be')
if st.button("Predict Next Word"):
    next_word = predict_next_word(model,tokenizer,input_text)
    st.write(f"Next Word : {next_word}")

