# import required packages
import warnings
from src.utils import create_dataset, denoise, remove_special_char, lemmatize_and_remove_stopwords, tokenize_test_data
import os
import pandas as pd
import pickle
from tensorflow import keras
warnings.filterwarnings('ignore')


if __name__ == "__main__":

	# Create the test dataset

	# Get the negative sentiment test data
	dataset_neg = create_dataset(r"\..\data\aclImdb\test\neg", 0)

	# Get the positive sentiment test data
	dataset_pos = create_dataset(r"\..\data\aclImdb\test\pos", 1)

	# Combine the negative and positive sentiment test data into one dataset
	dataset = dataset_neg + dataset_pos

	# Create dataframe and randomize the dataset
	dataset = pd.DataFrame(dataset, columns=['id', 'rating', 'review', 'target'])
	test_dataset = dataset.sample(frac=1)

	# Save the created dataset into a CSV file (Comment the following 5 code lines if a separate file is not required)
	dir_path = os.path.dirname(__file__)
	path = dir_path + r"\data\test_data_NLP.csv"
	test_dataset.to_csv(path, index = False)
	print("Dataset created")

	# Read back the dataset
	test_dataset = pd.read_csv(path)
	

	# Text preprocessing

	# Denoise the review column
	test_dataset['review'] = test_dataset['review'].apply(denoise)

	# Remove special characters from rows of review column
	test_dataset['review'] = test_dataset['review'].apply(remove_special_char)

	# Lemmatize text and remove stopwords
	review_text_list = lemmatize_and_remove_stopwords(test_dataset['review'])

	test_dataset.drop(columns=['review'])
	test_dataset['review'] = review_text_list

	# Load tokenizer object
	tokenizer_file = open(dir_path + r"\models\tokenizer_NLP.pkl", 'rb')
	tokenizer = pickle.load(tokenizer_file)
	tokenizer_file.close()

	# Tokenize and pad the input data
	reviews = tokenize_test_data(test_dataset['review'], tokenizer)

	rating = test_dataset['rating'].astype(int)
	dataset = pd.concat([reviews, rating, test_dataset['target']], axis = 1)

	# Save the dataset after preprocessing (Comment the following 4 code lines if a separate file is not required)
	dataset.to_csv(dir_path + r"\data\test_data_preprocessed_NLP.csv", index = False)
	print("Test dataset after preprocessing created")

	# Load data after preprocessing	
	path = dir_path + r"\data\test_data_preprocessed_NLP.csv"
	dataset = pd.read_csv(path)

	X_test = dataset.iloc[:,:-2]
	y_test = dataset.iloc[:,-1:]

	# Load the saved model
	path = dir_path + r"\models\Group81_NLP_model.h5"
	model = keras.models.load_model(path)
	print("Model loaded")

	# Evaluate the model on the test data
	results = model.evaluate(X_test, y_test)

	# Print the testing accuracy
	print("The testing accuracy is: {}".format(results[1]))