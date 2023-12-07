"""
Module for evaluating different text classification approaches.

This module orchestrates the comparison of three text classification approaches: Online Approach, Ngrams Approach,
and Word Embeddings Approach. It includes functions to load, preprocess, transform data, and calculate model accuracy.
"""

import pandas as pd
import os
import online_approach
import Ngrams_approach
import WordEmbeddings_approach

# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------
Online_Approach_Model = True
Ngrams_approach_Model = True
WordEmbeddings_Model = True

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

Dataset_Root = '../data'
Path_To_Dataset = os.path.join(Dataset_Root, 'chatgpt_paraphrases_modified.csv')
Figures_Root = '../figures'
Online_plot = os.path.join(Figures_Root, 'Online_plot.png')
Ngrams_plot = os.path.join(Figures_Root, 'Ngrams_plot.png')
WordEmbeddings_plot = os.path.join(Figures_Root, 'WordEmbeddings_plot.png')


# -----------------------------------------------------------------------------
# Exisiting Project

# This project has been already implemented and the link for the project is below.
# https://www.analyticsvidhya.com/blog/2023/04/how-to-build-a-machine-learning-model-to-distinguish-if-its-human-or-chatgpt/
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Present Implementation

# Used Chatgpt and other websited to understand the ways to extract features and what NLP techniques can be used here
# https://chat.openai.com/share/1954395d-66ba-4d60-8bb7-68aec664fe23
#https://chat.openai.com/share/46d12112-3991-4539-9a17-e405f5c2ffff
# https://www.datacamp.com/blog/what-is-tokenization
# https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/#:~:text=N%2Dgrams%20are%20continuous%20sequences,(Natural%20Language%20Processing)%20tasks.
# -----------------------------------------------------------------------------


def extract_data(Dataset_path: str, model: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset.

    Args:
    - Dataset_path : Path to the dataset.
    - model : Type of model ('Online', 'Ngrams', 'WordEmbedding').

    Returns:
    - pd.DataFrame: Processed dataset.
    """
    df = pd.read_csv(Dataset_path)
    if model == "Online":
        df = df.sample(n=5000)
    else:
        df = df.sample(n=125000)
    Dataset = transformations(df)
    return Dataset


def transformations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data transformations.

    Args:
    - data : Input data.

    Returns:
    - pd.DataFrame: Transformed data.
    """
    category = {}
    for i in range(len(data)):
        chatgpt = data.iloc[i]["paraphrases"][1:-1].split(', ')
        for j in chatgpt[:1]:
            category[j[1:-1]] = 'chatgpt'
        category[data.iloc[i]['text']] = "human"
    df = pd.DataFrame(category.items(), columns=["text", "category"])
    df = df.sample(frac=1)
    return df


def models_accuracy(df: pd.DataFrame, figure_path: str, model: str) -> None:
    """
    Calculate model accuracy.

    Args:
    - df : Input dataset.
    - figure_path : Path to save figures.
    - model : Type of model ('Online', 'Ngrams', 'WordEmbedding').

    Returns:
    - None
    """
    if model == "Online":
        online_approach_accuracy, f1_score, online_model_accuracy = online_approach.online(df, figure_path)
        print(online_approach_accuracy)
        print(f1_score)
        print(online_model_accuracy)
    elif model == "Ngrams":
        N_gram_accuracy = Ngrams_approach.Ngrams_approach(df, figure_path)
        print(N_gram_accuracy)
    else:
        WordEmbeddings_accuracy = WordEmbeddings_approach.word_embedding(df, figure_path)
        print(WordEmbeddings_accuracy)


# -----------------------------------------------------------------------------
# Script initialization
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if Online_Approach_Model:
        dataset = extract_data(Dataset_path=Path_To_Dataset, model="Online")
        models_accuracy(dataset, figure_path=Online_plot, model="Online")
    if Ngrams_approach_Model:
        dataset = extract_data(Dataset_path=Path_To_Dataset, model="Ngrams")
        models_accuracy(dataset, figure_path=Ngrams_plot, model="Ngrams")
    if WordEmbeddings_Model:
        dataset = extract_data(Dataset_path=Path_To_Dataset, model="WordEmbedding")
        models_accuracy(dataset, figure_path=WordEmbeddings_plot, model="WordEmbedding")
