# import required packages
import os
import glob
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Function to create datasets
def create_dataset(file_path, target):
    dir_path = os.path.dirname(__file__)

    dataset = []

    path = dir_path + os.path.join(file_path)

    # Read each text file from the given dataset
    for filename in glob.iglob(f'{path}\*'):
        row = []
        details = filename.replace(path,'').replace('.txt','')
        details = details.split('_')
        row.append(details[0]) # id
        row.append(details[1]) # rating
        
        with open(filename, encoding="utf8") as f:
            text = f.readlines()
            row.append(text) # review
            
        row.append(target) # target sentiment
        
        dataset.append(row)

    return dataset

# Function to strip HTML phrases
def html_strip(text):
    t = BeautifulSoup(text, "html.parser")
    return t.get_text()

# Function to remove noise (HTML phrases and extra braces)
def denoise(text):
    text = text.replace("[","").replace("]","")
    text = html_strip(text)
    return text

# Function to remove other special characters
def remove_special_char(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

# Function to lemmatize words and remove the stopwords
def lemmatize_and_remove_stopwords(data):
    lemmatizer = WordNetLemmatizer()
    review_text_list = []

    for i in range(len(data)):
        words = nltk.word_tokenize(data[i])
        words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
        review_text_list.append(' '.join(words))
        
        if i % 10000 == 0:
            print('Lemmatized row: ',i)
            
    review_text_list = pd.DataFrame(review_text_list, columns=['review'])

    return review_text_list

# Function to tokenize the training data and pass the tokenizer object
def tokenize(data):
    tokenizer = Tokenizer(num_words=2000, split=' ')
    tokenizer.fit_on_texts(data.values)
    X = tokenizer.texts_to_sequences(data.values)
    X = pad_sequences(X, maxlen=250)

    X = pd.DataFrame(X)  
    return X, tokenizer

# Function to tokenize the testing data using the tokenizer object trained on the training data
def tokenize_test_data(data, tokenizer):
    X = tokenizer.texts_to_sequences(data.values)
    X = pad_sequences(X, maxlen=250)

    X = pd.DataFrame(X)
    return X