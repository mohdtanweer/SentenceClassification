import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from sklearn.metrics import f1_score

# Module-level global variables
EMBEDDING_DIM = 100
LABEL_MAPPING = "../artifacts/main_product_label_mapping.csv"
GLOVE_DIR = "../data/glove.6B.100d.txt"

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # return f1_score(labels_flat, preds_flat, average=None)
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    mapping_df = pd.read_csv(LABEL_MAPPING)
    label_dict = dict(zip(mapping_df.main_product, mapping_df.label))
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def get_main_product(idx):
    mapping_df = pd.read_csv(LABEL_MAPPING)
    label_dict = dict(zip(mapping_df.main_product, mapping_df.label))
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    return label_dict_inverse[idx]

def embedding_matrix_glove(word_index):
    embeddings_index = {}
    f = open(GLOVE_DIR)
    logging.info(f'Loading GloVe from: {GLOVE_DIR}...')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    logging.info("Done.\n Proceeding with Embedding Matrix...")

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_model(vocab_size, embedding_dim, max_length, embedding_matrix):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='softmax')
    ])
    return model
