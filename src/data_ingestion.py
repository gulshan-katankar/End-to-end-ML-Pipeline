import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


#ensure the log dir exists 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True) #makedirs is used to make file dir and exist_ok ensures if its already there no overwriting should be done 


#logging configuration
logger = logging.getLogger('data ingestion') #logger name is data ingestion
logger.setLevel('DEBUG') #logger levels : debug, info, warning, error, critical

#console handler is used when we want to show logging info in the main terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#file handler is used when we want to show logging info in a seperate file 
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #formatter is used to define the format in which we want our logging messages 
#we set the same formatting for the console and file logger here 
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str)-> dict:
    """
    loads params from a yaml file

    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('params recieved from %s', params_path)
        return params
    except FileNotFoundError :
        logger.error('file not founf: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error: %s', e)
        raise
    except Exception as e:
        logger.error('unexpected error: %s', e)
        raise



def load_data(data_url: str) -> pd.DataFrame: #loads the data from url
    try:
        df = pd.read_csv(data_url)
        logger.debug('data looked from %s', data_url) #tells the data has been loaded 
        return df
    #defining exceptions :
    except pd.errors.ParserError as e :
        logger.error('failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    try:
        df.drop(columns= ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target','v2': 'text'}, inplace=True)
        logger.debug('data preprocesssing is complete')
        return df
    except KeyError as e:
        logger.error('missing column in dataframe: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data:pd.DataFrame, data_path:str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True) #we will make a subfolder named raw
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('unexpected error occured while saving the data : %s',e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        #test_size = 0.2 #old hard method
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df = load_data(data_url=data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('failed to complete the data ingestion process: %s', e)
        print(f"error : {e}")

if __name__ == '__main__':
    main()

    