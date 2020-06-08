import sys, getopt
import yaml
from pathlib import Path
import codecs
import pandas as pd
import psycopg2 as pg
import logging

from train import train_model
from predict import predict_class

# Module-level global variables
DATA_PATH = "../data"
CONFIG_PATH = "../misc/db_config.yaml"

logging.basicConfig(level=logging.DEBUG)

class DBConnection:
    def __init__(self, db_config_file):
        with open(db_config_file) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.db_config = config.get("pg")

    def __enter__(self):
        logging.info("Creating DB connection...")
        self.connection = pg.connect(
            host=self.db_config.get("host"),
            port=int(self.db_config.get("port")),
            dbname=self.db_config.get("dbname"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),

        )
        logging.info("Connection created!")
        return self.connection

    def __exit__(self, type, value, traceback):
        logging.info("Closing the DB connection!")
        self.connection.close()

class LoadTrainingData:
    def __init__(self, db_config_path):
        self.db_config_path = db_config_path
    
    def load_data_from_tables(self):
        with DBConnection(self.db_config_path) as connection:
            cur = connection.cursor()
            users = 'complaints_users'
            products = 'products'
            sql = f'select complaint_id, complaint_text, a.product_id, b.main_product \
from {users} a inner join {products} b on b.product_id=a.product_id;'
            try:
                df = pd.read_sql(sql, connection)
                return df
            except Exception as e:
                logging.error(f"Failed to load data from tables: {users}, {products}:", e)
                raise

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"htp:",["train","predict="])
    except getopt.GetoptError:
        logging.info('Incorrect argument... \nmain.py -t or --train to train the model || main.py -p or --predict <Text> to predict on model output')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info('main.py -t or --train to train the model || main.py -p or --predict <Text> to predict on model output')
            sys.exit()
        elif opt in ("-t", "--train"):
            logging.info("training model started...")
            get_data = LoadTrainingData(CONFIG_PATH)
            df = get_data.load_data_from_tables()
            logging.info(f"training data loaded... {df.shape}")
            train_model(df)
            logging.info("training model finished!")
        elif opt in ("-p", "--predict"):
            text = arg
            if len(text) > 10:
                main_prouct = predict_class(text)
                logging.info("Prediction done!")
            else:
                logging.info("too small sentence to predict")     

if __name__ == '__main__':
    main(sys.argv[1:])