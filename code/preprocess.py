import string
import json
import re
import nltk
import logging
import pandas as pd
import numpy as np

from nltk.corpus import stopwords

# Module-level global variables
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
LABEL_MAPPING = "../artifacts/main_product_label_mapping.csv"

# Converting the text into lower case
def clean_text(text):
    text = text.lower()
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\d+)"," ",text)
    text = re.sub("xxxx+", "", text)
    for word in STOPWORDS:
        token = " " + word + " "
        text = text.replace(token, " ")
    text = re.compile(" +").sub(" ", text)
    return text

def create_label(df):
    logging.info("creating labels for main products")
    main_prd_dict = {}
    for idx, label in enumerate(df.main_product.unique()):
        main_prd_dict[label] = idx
    df['label'] = df.main_product.replace(main_prd_dict)
    logging.info("saving the mapping of main products and labels in artifacts")
    mapping_df = pd.DataFrame(list(main_prd_dict.items()), columns=['main_product', 'label'])
    mapping_df.to_csv(LABEL_MAPPING, index=False)
    return df, main_prd_dict