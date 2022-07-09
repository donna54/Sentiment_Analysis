# import required packages
import warnings
from src.utils import create_dataset, denoise, remove_special_char, lemmatize_and_remove_stopwords, tokenize
import os
import pandas as pd
import nltk
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
import pickle
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


if __name__ == "__main__":

	# Create the train dataset

	# Get the negative sentiment train data
	dataset_neg = create_dataset(r"\..\data\aclImdb\train\neg", 0)

	# Get the positive sentiment train data
	dataset_pos = create_dataset(r"\..\data\aclImdb\train\pos", 1)

	# Combine the negative and positive sentiment train data into one dataset
	dataset = dataset_neg + dataset_pos

	# Create dataframe and randomize the dataset
	dataset = pd.DataFrame(dataset, columns=['id', 'rating', 'review', 'target'])
	train_dataset = dataset.sample(frac=1)

	# Save the created dataset into a CSV file (Comment the following 5 code lines if a separate file is not required)
	dir_path = os.path.dirname(__file__)
	path = dir_path + r"\data\train_data_NLP.csv"
	train_dataset.to_csv(path, index = False)
	print("Dataset created")

	# Read back the dataset
	train_dataset = pd.read_csv(path)


	# Text preprocessing

	# Denoise the review column
	train_dataset['review'] = train_dataset['review'].apply(denoise)

	# Remove special characters from rows of review column
	train_dataset['review'] = train_dataset['review'].apply(remove_special_char)

	# Lemmatize text and remove stopwords
	review_text_list = lemmatize_and_remove_stopwords(train_dataset['review'])

	train_dataset.drop(columns=['review'])
	train_dataset['review'] = review_text_list

	# Tokenize the text in the review column
	reviews, tokenizer = tokenize(train_dataset['review'])

	# Save the tokenizer object to use on test data
	pickle.dump(tokenizer, open(dir_path + r"\models\tokenizer_NLP.pkl", 'wb'))

	rating = train_dataset['rating'].astype(int)
	dataset = pd.concat([reviews, rating, train_dataset['target']], axis = 1)

	# Save the dataset after preprocessing (Comment the following 4 code lines if a separate file is not required)
	dataset.to_csv(dir_path + r"\data\train_data_preprocessed_NLP.csv", index = False)
	print("Train dataset after preprocessing created")

	# Load data after preprocessing	
	path = dir_path + r"\data\train_data_preprocessed_NLP.csv"
	dataset = pd.read_csv(path)

	X_train = dataset.iloc[:,:-2]
	y_train = dataset.iloc[:,-1:]


	# Build the model
	model = Sequential()

	model.add(Embedding(2000, 32, input_length = X_train.shape[1]))

	model.add(Dropout(0.2))

	model.add(LSTM(100))
	model.add(Dense(256, activation='relu'))

	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

	# Define the batch size
	batch_size = 64

	# Fit the model over the training data
	history = model.fit(X_train, y_train, epochs = 5, batch_size = batch_size, verbose = 'auto', validation_split=0.1)

	# Save the model
	model.save(dir_path + r"\models\Group81_NLP_model.h5")

	# Print the final training accuracy
	print('The final training accuracy is: {}'.format(history.history['accuracy'][-1]))