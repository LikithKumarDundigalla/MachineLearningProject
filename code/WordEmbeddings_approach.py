import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def data_transformations(df):
    df.dropna(inplace=True)
    text_data = df[df['text'].apply(lambda x: isinstance(x, str))]
    human_data = text_data[text_data['category'] == 'human']['text']
    ai_data = text_data[text_data['category'] == 'chatgpt']['text']
    all_data = pd.concat([human_data, ai_data], axis=0)
    labels = [0] * len(ai_data) + [1] * len(human_data)
    return all_data, labels


def get_sentence_vector(model, sentence):
    word_vectors = [model.wv[word] for word in sentence.split() if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def ML_model(sentence_vectors, labels):
    X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, labels, test_size=0.2, random_state=42)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    # Predict on test data and evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    return accuracy, y_test, y_pred


def plot(y_test, y_pred, figure_path):
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Plotting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['AI-generated', 'Human-generated'],
                yticklabels=['AI-generated', 'Human-generated'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(figure_path)


def word_embedding(df, figure_path):
    print("In Word Embedding")
    all_data, labels = data_transformations(df)
    print("Word Embedding Transformation Done")
    model = Word2Vec(all_data, vector_size=100, window=5, min_count=1, workers=4)
    print("Word Embedding Model Done")
    sentence_vectors = [get_sentence_vector(model, sentence) for sentence in all_data]
    print("Word Embedding Sentence Vectors Done")
    accuracy, y_test, y_pred = ML_model(sentence_vectors, labels)
    print("Word Embedding accuracy Done")
    plot(y_test, y_pred,figure_path)
    return accuracy
