import os
import numpy as np
import pandas as pd
import pickle #to save the model
import logging
from sklearn.ensemble import RandomForestClassifier

#ensure the "logs" dir exists 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging config
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str)-> pd.DataFrame:
    """
    load data from a csv file
    
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: DataFrame

    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('file not found: %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loadin the data: %s', e)
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params:dict)-> RandomForestClassifier:
    """
    train the random forest model
    
    :param x_train: training features
    :type x_train: np.ndarray
    :param y_train: training labels 
    :type y_train: np.ndarray
    :param params: Dictionary of hyperparams 
    :type params: dict
    :return: trained random forest classifier 
    :rtype: RandomForestClassifier
    """

    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("the number of samples in x_train and y_train must be the same")
        
        logger.debug('initialize the random forest with params: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug('model training started with %d samples',x_train.shape[0])
        clf.fit(x_train, y_train)
        logger.debug('model training completed')
        
        return clf
    except ValueError as e:
        logger.error('value error during model training : %s', e)
        raise
    except Exception as e:
        logger.error('error during model training: %s',e)
        raise

def save_model(model, file_path: str)-> None:
    """
    saved the trained model to a file
    
    :param model: trained model object
    :param file_path: path to save the model
    :type file_path: str
    """

    try:
        #ensure the dir exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("model saved to : %s", file_path)
    except FileNotFoundError as e:
        logger.error('file path not found: %s', e)
        raise
    except Exception as e:
        logger.error("error occured while saving the model : %s", e)
        raise


def main():
    try:
        params = {'n_estimators': 25, 'random_state':2}
        train_data = load_data('./data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('failed to complete the model building process : %s', e)
        print(f"error: {e}")


if __name__ == '__main__':
    main()
