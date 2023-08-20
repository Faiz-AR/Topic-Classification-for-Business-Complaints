import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Verify if NLTK library tools is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(len(df)):
        try:
            processed_complaint = remove_stopwords_and_tokenize(df['complaint'].loc[i])
            complaints = lemmatize_and_concat(processed_complaint)
            df['complaint'].loc[i] = complaints
        except KeyError:
            continue
                
    return df
    
def remove_stopwords_and_tokenize(complaint: str) -> list:
    stopwords_list = stopwords.words('english') + list(string.punctuation) + [
        "''", '""', '...', '``', '--', 'xxxx', 'xxxxxxxx']
    
    tokens = nltk.word_tokenize(complaint)
    stopwords_removed = [token for token in tokens if token.lower() not in stopwords_list]
    
    # Remove all tokens with numbers and punctuation
    stopwords_punc_and_numbers_removed = [word.lower() for word in stopwords_removed if word.isalpha()]
    
    return stopwords_punc_and_numbers_removed

def lemmatize_and_concat(list_of_words) -> str:
    lemmatizer = WordNetLemmatizer()
    
    # Remove any NaN's
    list_of_words = [i for i in list_of_words if i is not np.nan]
    
    # Lemmatize each word
    lemmatized_list = []
    for idx, word in enumerate(list_of_words):
        lemmatized_list.append(lemmatizer.lemmatize(word))
    
    # Make the list into a single string with the words separated by ' '
    concat_words = ''
    for word in lemmatized_list:
        concat_words += word + ' '
    return concat_words.strip()
    