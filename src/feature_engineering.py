import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

#ensure the logs directory exists 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging config
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str)-> dict:
    """
    load params from the yaml file

    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('parameters are loaded from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('file not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error: %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error occured : %s', e)
        raise


def load_data(file_path: str)->pd.DataFrame: #loadd the data fro interim folder
    """
    load data from csv file
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('data loaded and nans filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file : %s',e)
    except Exception as e:
        logger.error('unexpected error occured while loading the data: %s', e)
        raise


def apply_tfid(train_data: pd.DataFrame, test_data:pd.DataFrame, max_features:int)->tuple:
    """
    Docstring for apply_tfid
    
    :param train_data: Description
    :type train_data: pd.DataFrame
    :param test_data: Description
    :type test_data: pd.DataFrame
    :param max_features: Description
    :type max_features: int
    :return: Description
    :rtype: tuple

    apply tfid to the data
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('tfid applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('error during bag of words transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path:str)-> None:
    """
    Docstring for save_data
    
    :param df: Description
    :type df: pd.DataFrame
    :param file_path: Description
    :type file_path: str

    save the dataframe to csv file 
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('data saved to  %s, file_path')
    except Exception as e:
        logger.error('unexpected error occured while saving the data: %s ', e)
        raise

def main():
    try: 
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        #max_features = 50 #max no of cols  u will see in feature engineered dataset

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfid(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data","processed","train_tfidf.csv"))
        save_data(test_df, os.path.join("./data","processed", "test_tfid.csv"))
    except Exception as e:
        logger.error('failed to complete the feature engineering process: %s', e)
        print(f"error: {e}")

if __name__ == '__main__':
    main()