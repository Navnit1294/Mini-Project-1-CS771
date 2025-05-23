import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load datasets (adjust file paths if needed)
train_emoticon = pd.read_csv('train_emoticon.csv')
train_text_seq = pd.read_csv('train_text_seq.csv')
train_feature = np.load('train_feature.npz', allow_pickle=True)

valid_emoticon = pd.read_csv('valid_emoticon.csv')
valid_text_seq = pd.read_csv('valid_text_seq.csv')
valid_feature = np.load('valid_feature.npz', allow_pickle=True)

# Extract labels
y_train = train_emoticon['label'].astype(int)
y_valid = valid_emoticon['label'].astype(int)

# One-hot encode emoticons with handle_unknown='ignore'
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_emoticon_encoded = encoder.fit_transform(train_emoticon.drop(columns=['label']))
X_valid_emoticon_encoded = encoder.transform(valid_emoticon.drop(columns=['label']))

# Convert text sequences into numerical form using CountVectorizer (character-based)
vectorizer = CountVectorizer(analyzer='char')
X_train_text_seq_vectorized = vectorizer.fit_transform(train_text_seq['input_str'])
X_valid_text_seq_vectorized = vectorizer.transform(valid_text_seq['input_str'])

# Load and bin deep features
X_train_deep_feat = train_feature['features']
X_valid_deep_feat = valid_feature['features']

# Binning the deep features to make them categorical for MultinomialNB
n_bins = 10
X_train_deep_feat_binned = np.digitize(X_train_deep_feat, bins=np.linspace(X_train_deep_feat.min(), X_train_deep_feat.max(), n_bins))
X_valid_deep_feat_binned = np.digitize(X_valid_deep_feat, bins=np.linspace(X_valid_deep_feat.min(), X_valid_deep_feat.max(), n_bins))

# Combine emoticons, text sequences, and binned deep features
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

# Define percentages of training data to use and range of alpha values
percentages = [0.2, 0.4, 0.6, 0.8, 1.0]
alpha_values = [0.1, 0.5, 1.0, 1.5, 2.0]
accuracy_results = {}

# Initialize variables to track the best accuracy and its corresponding parameters
best_accuracy = 0
best_alpha = None
best_percentage = None

# Iterate through different alpha values
for alpha in alpha_values:
    accuracies = []
    for percent in percentages:
        # Determine the number of samples to use from the training set
        n_samples = int(len(X_train_combined) * percent)
        
        # Train the model on the specified percentage of training data with the given alpha
        nb_model = MultinomialNB(alpha=alpha)
        nb_model.fit(X_train_combined[:n_samples], y_train[:n_samples])
        
        # Make predictions on the validation set
        y_pred = nb_model.predict(X_valid_combined)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_valid, y_pred)
        accuracies.append(accuracy)
        
        # Update the best accuracy and parameters if current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha
            best_percentage = percent
        
        print(f'Validation Accuracy (alpha={alpha}, Training {int(percent * 100)}%): {accuracy:.4f}')
    
    # Store the accuracies for each alpha
    accuracy_results[alpha] = accuracies

# Plotting accuracy curves for different alphas
plt.figure(figsize=(12, 8))
for alpha, accuracies in accuracy_results.items():
    plt.plot([int(p * 100) for p in percentages], accuracies, marker='o', label=f'alpha={alpha}')
    
plt.title('Accuracy vs. Percentage of Training Data for Different Alpha Values')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Validation Accuracy')
plt.xticks([20, 40, 60, 80, 100])
plt.ylim(0, 1)
plt.legend(title='Alpha values')
plt.grid()
plt.show()

# Print the best alpha, percentage, and corresponding accuracy
print(f'\nBest Validation Accuracy: {best_accuracy:.4f}')
print(f'Best Alpha: {best_alpha}')
print(f'Best Percentage of Training Data: {int(best_percentage * 100)}%')

#This script trains a Multinomial Naive Bayes (MultinomialNB) classifier using combined features from three different sources: emoticon one-hot encodings, character-based text sequences, and discretized deep features. The model is trained on 40% of the training data, with an alpha value of 0.1 for smoothing.