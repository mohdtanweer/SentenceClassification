import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import pickle

from preprocess import clean_text
from utils import embedding_matrix_glove, get_main_product, create_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Module-level global variables
GLOVE_DIR = "../data/glove.6B.100d.txt"
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 100
MAX_LENGTH = 200
NUM_EPOCHS = 10
BATCH_SIZE=256
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = '<OOV>'
TOKENIZER_PATH = "../artifacts/tokenizer.pickle"
MODEL = "../artifacts/model.h5"

def load_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer

def predict_class(text):
    logging.info(f"Input text: {text}")
    logging.info("cleaning input text")
    text = clean_text(text)
    sentence = [text]
    tokenizer = load_tokenizer()
    logging.info("trained tokenizer loaded")
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    text_sequences = tokenizer.texts_to_sequences(sentence)
    text_padded = pad_sequences(text_sequences, padding=PADDING_TYPE, truncating=TRUNC_TYPE,maxlen=MAX_LENGTH)

    logging.info("creating embedding matrix using Glove Embeddings")
    embedding_matrix = embedding_matrix_glove(word_index)
    logging.info(f"Embeddings Weights created {embedding_matrix.shape}")

    logging.info("getting pre-trained model")
    model = create_model(vocab_size, EMBEDDING_DIM, MAX_LENGTH, embedding_matrix)
    print(model.summary())
    logging.info("loading model weights")
    model.load_weights(MODEL)

    predict = model.predict(text_padded)
    predict = np.argmax(predict)

    predicted_main_product = get_main_product(predict)
    logging.info(f'Predicted Main Product: {predicted_main_product}')
