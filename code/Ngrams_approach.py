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


def data_transformations(df):
    text_data = df[df['text'].apply(lambda x: isinstance(x, str))]
    human_data = text_data[text_data['category'] == 'human']['text']
    ai_data = text_data[text_data['category'] == 'chatgpt']['text']
    all_data = pd.concat([human_data, ai_data], axis=0)
    labels = ['human'] * len(human_data) + ['AI'] * len(ai_data)
    return all_data, labels


def token_data(all_data):
    tokenized_data = [' '.join(nltk.word_tokenize(str(sentence).lower())) for sentence in all_data if
                      isinstance(sentence, str)]
    return tokenized_data


def ngram_vectorizer(tokenized_data):
    ngram_vectorizer = CountVectorizer(ngram_range=(1, 2))
    ngram_features = ngram_vectorizer.fit_transform(tokenized_data)
    return ngram_features


def model(ngram_features, labels):
    X_train, X_test, y_train, y_test = train_test_split(ngram_features, labels, test_size=0.2, random_state=42)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict on test set
    predictions = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions) * 100
    report = classification_report(y_test, predictions)
    return accuracy, report, y_test, predictions


def plot_graph(y_test, predictions, figure_path):
    conf_matrix = confusion_matrix(y_test, predictions)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=['human', 'AI'],
                yticklabels=['human', 'AI'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(figure_path)


def Ngrams_approach(df,figure_path):
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
