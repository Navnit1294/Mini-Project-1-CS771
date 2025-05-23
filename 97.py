import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import KBinsDiscretizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

### Emoticon Model ###
# Convert each emoticon string to Unicode values
def emoticons_to_unicode(emoticon_string):
    return [ord(char) for char in emoticon_string]

# Load datasets
train_data = pd.read_csv('train_emoticon.csv')
valid_data = pd.read_csv('valid_emoticon.csv')
test_emoticon_data = pd.read_csv('test_emoticon.csv')

# Extract features and labels
X_train = train_data.iloc[:, :-1].squeeze().apply(emoticons_to_unicode).tolist()
y_train = train_data.iloc[:, -1]
X_valid = valid_data.iloc[:, :-1].squeeze().apply(emoticons_to_unicode).tolist()
y_valid = valid_data.iloc[:, -1]

# One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train)
X_valid_encoded = encoder.transform(X_valid)

# Train model
clf = MultinomialNB(alpha=1)
clf.fit(X_train_encoded, y_train)

# Prepare test data and predict
X_test_emoticon = test_emoticon_data.iloc[:, :].squeeze().apply(emoticons_to_unicode).tolist()
X_test_emoticon_encoded = encoder.transform(X_test_emoticon)
y_test_emoticon_pred = clf.predict(X_test_emoticon_encoded)

# Save test predictions
with open('pred_emoticon.txt', 'w') as f:
    for pred in y_test_emoticon_pred:
        f.write(f"{pred}\n")

### Combined Feature Model (Emoticons, Text, and Deep Features) ###
# Load datasets
train_emoticon = pd.read_csv('train_emoticon.csv')
train_text_seq = pd.read_csv('train_text_seq.csv')
train_feature = np.load('train_feature.npz', allow_pickle=True)

valid_emoticon = pd.read_csv('valid_emoticon.csv')
valid_text_seq = pd.read_csv('valid_text_seq.csv')
valid_feature = np.load('valid_feature.npz', allow_pickle=True)

test_emoticon = pd.read_csv('test_emoticon.csv')
test_text_seq = pd.read_csv('test_text_seq.csv')
test_feature = np.load('test_feature.npz', allow_pickle=True)

# Extract labels
y_train = train_emoticon['label'].astype(int)
y_valid = valid_emoticon['label'].astype(int)

# One-hot encode emoticons
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_emoticon_encoded = encoder.fit_transform(train_emoticon.drop(columns=['label']))
X_valid_emoticon_encoded = encoder.transform(valid_emoticon.drop(columns=['label']))
X_test_emoticon_encoded = encoder.transform(test_emoticon)

# Vectorize text sequences
vectorizer = CountVectorizer(analyzer='char')
X_train_text_seq_vectorized = vectorizer.fit_transform(train_text_seq['input_str'])
X_valid_text_seq_vectorized = vectorizer.transform(valid_text_seq['input_str'])
X_test_text_seq_vectorized = vectorizer.transform(test_text_seq['input_str'])

# Bin deep features
n_bins = 10
X_train_deep_feat = train_feature['features']
X_valid_deep_feat = valid_feature['features']
X_test_deep_feat = test_feature['features']

# Ensure deep features are 2D
X_train_deep_feat = X_train_deep_feat.reshape(X_train_deep_feat.shape[0], -1)
X_valid_deep_feat = X_valid_deep_feat.reshape(X_valid_deep_feat.shape[0], -1)
X_test_deep_feat = X_test_deep_feat.reshape(X_test_deep_feat.shape[0], -1)

# Bin deep features
binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_train_deep_feat_binned = binner.fit_transform(X_train_deep_feat)
X_valid_deep_feat_binned = binner.transform(X_valid_deep_feat)
X_test_deep_feat_binned = binner.transform(X_test_deep_feat)

# Combine features for training, validation, and test sets
X_train_combined = np.hstack([
    X_train_emoticon_encoded.toarray(),
    X_train_text_seq_vectorized.toarray(),
    X_train_deep_feat_binned.reshape(X_train_deep_feat_binned.shape[0], -1)
])
X_valid_combined = np.hstack([
    X_valid_emoticon_encoded.toarray(),
    X_valid_text_seq_vectorized.toarray(),
    X_valid_deep_feat_binned.reshape(X_valid_deep_feat_binned.shape[0], -1)
])
X_test_combined = np.hstack([
    X_test_emoticon_encoded.toarray(),
    X_test_text_seq_vectorized.toarray(),
    X_test_deep_feat_binned.reshape(X_test_deep_feat_binned.shape[0], -1)
])

# Train model
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_combined, y_train)

# Predict on test set
y_test_pred = nb_model.predict(X_test_combined)
with open('pred_combined.txt', 'w') as f:
    for pred in y_test_pred:
        f.write(f"{pred}\n")

### Deep Feature Model ###
train_data = np.load('train_feature.npz')
valid_data = np.load('valid_feature.npz')
test_data = np.load('test_feature.npz')

X_train = train_data['features']
y_train = train_data['label']
X_valid = valid_data['features']
y_valid = valid_data['label']
X_test = test_data['features']

# Ensure deep features are 2D
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Bin deep features
binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_train_binned = binner.fit_transform(X_train)
X_valid_binned = binner.transform(X_valid)
X_test_binned = binner.transform(X_test)

# Train model
nb_model_deep = MultinomialNB()
nb_model_deep.fit(X_train_binned, y_train)

# Predict on test set
y_test_pred_deep = nb_model_deep.predict(X_test_binned)
with open('pred_deep.txt', 'w') as f:
    for pred in y_test_pred_deep:
        f.write(f"{pred}\n")

### Text Sequence Model (RNN) ###
def preprocess_sequences(X_data, max_len=50):
    X_sequences = [[int(digit) for digit in seq] for seq in X_data]
    return pad_sequences(X_sequences, maxlen=max_len, padding='post')

# Load datasets
train_text_seq = pd.read_csv('train_text_seq.csv')
valid_text_seq = pd.read_csv('valid_text_seq.csv')
test_text_seq = pd.read_csv('test_text_seq.csv')

# Prepare the data
X_train_full = train_text_seq['input_str'].astype(str)
y_train_full = train_text_seq['label'].astype(int)
X_valid = valid_text_seq['input_str'].astype(str)
y_valid = valid_text_seq['label'].astype(int)
X_test = test_text_seq['input_str'].astype(str)

# Preprocess sequences
X_train_full_prep = preprocess_sequences(X_train_full)
X_valid_prep = preprocess_sequences(X_valid)
X_test_prep = preprocess_sequences(X_test)

# Convert labels to categorical
y_train_full_cat = to_categorical(y_train_full, num_classes=2)
y_valid_cat = to_categorical(y_valid, num_classes=2)

# Build RNN model
input_dim = np.max(X_train_full_prep) + 1
def build_rnn_model(input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=16))
    model.add(SimpleRNN(units=32, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train RNN model
rnn_model = build_rnn_model(X_train_full_prep.shape[1])
rnn_model.fit(X_train_full_prep, y_train_full_cat, epochs=10, batch_size=32, validation_data=(X_valid_prep, y_valid_cat))

# Predict on test set
y_test_pred_rnn = rnn_model.predict(X_test_prep)
y_test_pred_rnn_classes = np.argmax(y_test_pred_rnn, axis=1)

# Save test predictions
np.savetxt('pred_textseq.txt', y_test_pred_rnn_classes, fmt='%d')