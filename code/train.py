import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import pickle

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from preprocess import clean_text, create_label
from utils import create_model, f1_score_func, embedding_matrix_glove

# Module-level global variables
GLOVE_DIR = "../data/glove.6B.100d.txt"
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 100
MAX_LENGTH = 200
NUM_EPOCHS = 100
BATCH_SIZE=256
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = '<OOV>'
TOKENIZER_PATH = "../artifacts/tokenizer.pickle"
MODEL = "../artifacts/model.h5"

def handle_imbalance(df):
    # Handling data imbalance by removing product ids with less than 500 record counts
    # Creating a temporary dataframe with record counts of each product ids
    count_df = pd.DataFrame({'count': df.groupby(['product_id', 'main_product']).size()}).reset_index()
    # List of product ids with records cound less than 500
    product_ids = list(count_df[count_df['count'] < 500].product_id.values)
    df = df[~df.product_id.isin(product_ids)]
    return df

def train_val_test_split(df):
    df.set_index('complaint_id', inplace=True)
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        df.index.values,
        df.label.values,
        test_size=0.2,
        random_state=17,
        stratify=df.label.values
    )

    df_val_test = df.loc[X_val_test]
    X_val, X_test, y_val, y_test = train_test_split(
        df_val_test.index.values,
        df_val_test.label.values,
        test_size=0.5,
        random_state=17,
        stratify=df_val_test.label.values
    )
    df['data_type'] = ['not_set'] * df.shape[0]
    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    df.loc[X_test, 'data_type'] = 'test'
    return df

def encode(data):
    logging.info(f'Shape of data (BEFORE encode): {data.shape}')
    encoded = to_categorical(data)
    logging.info(f'Shape of data (AFTER  encode): {encoded.shape}')
    return encoded

def train_model(df):
    df = handle_imbalance(df)
    df.complaint_text = df.complaint_text.progress_map(clean_text)
    df, main_prd_dict = create_label(df)
    logging.info("labels created!")
    df = train_val_test_split(df)
    logging.info("train, validation and test set created!")
    print(df.head())
    train_sentences = df[df.data_type == 'train'].complaint_text.values
    validation_sentences = df[df.data_type == 'val'].complaint_text.values
    test_sentences = df[df.data_type == 'test'].complaint_text.values

    train_labels = df[df.data_type == 'train'].label.values
    validation_labels = df[df.data_type == 'val'].label.values
    test_labels = df[df.data_type == 'test'].label.values

    train_labels = encode(train_labels)
    validation_labels = encode(validation_labels)
    test_labels = encode(test_labels)
    logging.info("one hot encoding of labels done!")

    logging.info("calling tokenizer on training sentences")
    tokenizer = Tokenizer(num_words = MAX_NB_WORDS, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=PADDING_TYPE, truncating=TRUNC_TYPE,maxlen=MAX_LENGTH)
    logging.info("saving tokenizer into artifacts")
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    val_sequences = tokenizer.texts_to_sequences(validation_sentences)
    val_padded = pad_sequences(val_sequences,padding=PADDING_TYPE, truncating=TRUNC_TYPE,maxlen=MAX_LENGTH)

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences,padding=PADDING_TYPE, truncating=TRUNC_TYPE,maxlen=MAX_LENGTH)

    logging.info("creating embedding matrix using Glove Embeddings")
    embedding_matrix = embedding_matrix_glove(word_index)
    logging.info(f"Embeddings Weights created {embedding_matrix.shape}")

    logging.info("defining parameters for early stopping")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    ## We can also use a checkpoint call back to save the best model where the validation loss is minimum. For now, just skipping it ##
    
    model = create_model(vocab_size, EMBEDDING_DIM, MAX_LENGTH, embedding_matrix)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    print(model.summary())
    # logging.info("Model Summary: \n", model.summary())

    logging.info("starting model training ...")
    model.fit(train_padded,
        train_labels,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_padded, validation_labels),
        callbacks = [early_stopping]
    )
    logging.info("Training Done!")

    model.save_weights(MODEL)
    logging.info(f"model weights saved in {MODEL}")

    logging.info("evaluating f1 score of the model on test data")
    predictions = model.predict(test_padded)
    f1_score = f1_score_func(predictions, np.array(df[df.data_type == 'test'].label))
    logging.info(f"f1 score: {f1_score}")
