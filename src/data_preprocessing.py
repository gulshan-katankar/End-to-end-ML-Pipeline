import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

#making a log directory
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#setting up logger 
logger = logging.getLogger('data_preprocessing')#defining the name of logger 
logger.setLevel('DEBUG')

#setting up logger 
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_pth = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_pth)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #formatter is used to define the format in which we want our logging messages 
#we set the same formatting for the console and file logger here 
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text) : 
    """
    transforms the texts to lowercase, tokenizing, removing stopwords and punctuations and stemming 
    """

    ps = PorterStemmer()
    #convert to lower case
    text = text.lower()

    #tokenize the text 
    text = nltk.word_tokenize(text)

    #remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    #remove stopwords and punctuations 
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #stem the words
    text = [ps.stem(word) for word in text]

    #join the tokens back into a single string 
    return " ".join(text)


def preprocess_df (df, text_column = 'text', target_column = 'target'):
    """
    preprocess the dataframe by encoding the target col, removing duplicates and transforming the text column
    
    """

    try:
        logger.debug('starting preprocessing for data frame')
        #encode the target col
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('target col encoded')

        #remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('duplicated removed')

        #apply text transformations to the specified text col
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text col transformed')
        return df
    
    except KeyError as e:
        logger.error('col not found: %s', e)
        raise
    except Exception as e:
        logger.error('error during text normalisation: %s',e)
        raise

def main(text_column='text', target_column='target'): 
    """
    main function to load raw data, preprocess it and save the processed data
    """ 
    try:
        #fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('data loaded properly')

        #transform the data 
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        #store the data inside data/processed
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('preprocessed data saved in %s', data_path)
    except FileNotFoundError as e:
        logger.error('file not found %s',e)
    except pd.errors.EmptyDataError as e:
        logger.error('no data %s',e)
    except Exception as e:
        logger.error('failed to complete the data transformatiion process : %s', e)
        print(f"error: {e}")

if __name__ == '__main__':
    main()