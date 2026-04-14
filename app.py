import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences




#For deplyment
# Load the LSTM model and tokenizer
model = load_model('next_word_lstm.h5')

with open("tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)



# Get the max sequence length the model was trained on
max_sequence_len = model.input_shape[1]  # e.g. 17

def sample_with_temperature(preds, temperature=1.0):
    return np.random.choice(len(preds), p=preds)

def predict_next_word(model, tokenizer, text):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Fix 1: Guard against empty token list (unknown words)
    if len(token_list) == 0:
        return None

    # Fix 2: Pad to the correct fixed length the model expects
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len,
    )

    predicted = model.predict(token_list, verbose=0)[0]

    predicted_word_index = sample_with_temperature(predicted)

    return tokenizer.index_word.get(predicted_word_index, None)


## Streamlit app
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter the sequence of words")

if st.button("Predict Next Word"):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        next_word = predict_next_word(model, tokenizer, input_text)
        if next_word is None:
            st.error(
                "None of the entered words were found in the vocabulary. "
                "Try different words."
            )
        else:
            st.success(f"Next Word: **{next_word}**")