# -----------------------------------------------------------------------------
# Code for getting data from huggingface
#https://huggingface.co/datasets/humarin/chatgpt-paraphrases
# -----------------------------------------------------------------------------

from datasets import load_dataset
import pandas as pd

dataset = load_dataset('humarin/chatgpt-paraphrases')
train_data = dataset['train']
df = pd.DataFrame(train_data)
df = df.sample(n=150000)
df.to_csv('../data/chatgpt_paraphrases_modified.csv')