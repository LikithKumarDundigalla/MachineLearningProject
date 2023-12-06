"""
Module to perform text classification using N-grams approach.

This module includes functions for data transformations, tokenization, creating N-gram features,
building a classification model using Multinomial Naive Bayes, plotting a confusion matrix,
and executing the N-grams approach for text classification.
"""
import nltk

nltk.download('punkt')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def data_transformations(df: pd.DataFrame) -> tuple:
    """
    Transform the data to extract text and category labels.

    Args:
    - df : DataFrame containing 'text' and 'category' columns.

    Returns:
    - tuple: A tuple containing processed text data and corresponding labels.
    """
    # Extract text data and filter non-string entries
    text_data = df[df['text'].apply(lambda x: isinstance(x, str))]
    # Separate human and AI text data
    human_data = text_data[text_data['category'] == 'human']['text']
    ai_data = text_data[text_data['category'] == 'chatgpt']['text']
    # Concatenate text data and create labels
    all_data = pd.concat([human_data, ai_data], axis=0)
    labels = ['human'] * len(human_data) + ['AI'] * len(ai_data)
    return all_data, labels


def token_data(all_data: pd.Series) -> list:
    """
    Tokenize the text data.

    Args:
    - all_data : Series containing text data.

    Returns:
    - list: List of tokenized sentences.
    """
    # Tokenize sentences and convert to lowercase
    tokenized_data = [' '.join(nltk.word_tokenize(str(sentence).lower())) for sentence in all_data if
                      isinstance(sentence, str)]
    return tokenized_data


def ngram_vectorizer(tokenized_data: list) -> 'scipy.sparse.csr_matrix':
    """
    Vectorize the tokenized data into N-gram features.

    Args:
    - tokenized_data : List of tokenized sentences.

    Returns:
    - scipy.sparse.csr_matrix: N-gram features.
    """
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
    ngram_features = ngram_vectorizer.fit_transform(tokenized_data)
    return ngram_features


def model(ngram_features: 'scipy.sparse.csr_matrix', labels: list) -> tuple:
    """
    Build and evaluate the classification model using Multinomial Naive Bayes.

    Args:
    - ngram_features : N-gram features.
    - labels : List of labels.

    Returns:
    - tuple: Accuracy, classification report, true labels, and predicted labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(ngram_features, labels, test_size=0.2, random_state=42)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict on test set
    predictions = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions) * 100
    report = classification_report(y_test, predictions)
    return accuracy, report, y_test, predictions


def plot_graph(y_test: list, predictions: list, figure_path: str) -> None:
    """
    Plot and save the confusion matrix.

    Args:
    - y_test : True labels.
    - predictions : Predicted labels.
    - figure_path : Path to save the figure.
    """
    conf_matrix = confusion_matrix(y_test, predictions)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['human', 'AI'],
                yticklabels=['human', 'AI'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(figure_path)


def Ngrams_approach(df: pd.DataFrame, figure_path: str) -> float:
    """
    Execute the N-grams approach for text classification.

    Args:
    - df : DataFrame containing 'text' and 'category' columns.
    - figure_path : Path to save the plotted confusion matrix figure.

    Returns:
    - float: Accuracy of the model.
    """
    print("In Ngrams")
    all_data, labels = data_transformations(df)
    print("Ngrams Transformation Done")
    tokenized_data = token_data(all_data)
    print("Ngrams Tokenization Done")
    ngram_features = ngram_vectorizer(tokenized_data)
    print("Ngrams Features Done")
    accuracy, report, y_test, predictions = model(ngram_features, labels)
    print("Ngrams accuracy Done")
    plot_graph(y_test, predictions,figure_path)
    return accuracy
