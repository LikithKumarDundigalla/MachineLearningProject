import pandas as pd
import os
import online_approach
import Ngrams_approach
import WordEmbeddings_approach


# -----------------------------------------------------------------------------
# Control exercise execution
# -----------------------------------------------------------------------------
Online_Approach_Model = False
Ngrams_approach_Model = False
WordEmbeddings_Model = False

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

Dataset_Root = '../data'
Path_To_Dataset = os.path.join(Dataset_Root, 'chatgpt_paraphrases_modified.csv')
Figures_Root = '../figures'
Online_plot = os.path.join(Figures_Root, 'Online_plot.png')
Ngrams_plot = os.path.join(Figures_Root, 'Ngrams_plot.png')
WordEmbeddings_plot = os.path.join(Figures_Root, 'WordEmbeddings_plot.png')

def extract_data(Dataset_path, model):
    df = pd.read_csv(Dataset_path)
    if model == "Online":
        df = df.sample(n=4000)
    else:
        df = df.sample(n=100000)
    Dataset = transformations(df)
    return Dataset


def transformations(data):
    category = {}
    for i in range(len(data)):
        chatgpt = data.iloc[i]["paraphrases"][1:-1].split(', ')
        for j in chatgpt[:1]:
            category[j[1:-1]] = 'chatgpt'
        category[data.iloc[i]['text']] = "human"
    df = pd.DataFrame(category.items(), columns=["text", "category"])
    df = df.sample(frac=1)
    return df


def models_accuracy(df, figure_path, model):
    if model == "Online":
        online_approach_accuracy = online_approach.online(df, figure_path)
        print(online_approach_accuracy)
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
