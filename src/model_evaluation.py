import os
import numpy as np
import pandas as pd
import pickle 
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score,roc_auc_score
import logging
import yaml
from dvclive import Live


#ensure the logs dir exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging config
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str)->dict:
    """
    load params from a yaml file

    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('parameters loaded succesfully form %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('file not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error :%s', e)
        raise
    except Exception as e:
        logger.error('unexpected error : %s', e)
        raise


def load_model(file_path:str):
    """
    load the trained model from a file

    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('file not found : %s', file_path)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the model: %s', e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """
    load data from csv file
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data: %s', e)
        raise

def evaluate_model(clf, x_test:np.ndarray, y_test:np.ndarray)-> dict:
    """
    evaluate the model and return the evaluation metrics 

    """
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict ={
            'accuracy':accuracy,
            'precision':precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('model eval metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('error during model eval: %s', e)
        raise

def save_metrics(metrics:dict, file_path:str)-> None:
    """
    save the eval metrics to a json file
    """
    try:
        #ensure the dir exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('metrics saved to %s', file_path)
    except Exception as e:
        logger.error('error occured while saving the metrics')
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfid.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test)

        #experiment tracking using dvc live
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))

            live.log_params(params)

        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('failed to complete the model eval process : %s', e)
        print(f"error: {e}")

if __name__ == '__main__':
    main()
