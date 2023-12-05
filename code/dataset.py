from datasets import load_dataset
import pandas as pd


def load_and_process_data() -> pd.DataFrame:
    """
    Load and process data from the 'humarin/chatgpt-paraphrases' dataset.

    Returns:
    - pd.DataFrame: Processed DataFrame containing the dataset.
    """
    dataset = load_dataset('humarin/chatgpt-paraphrases')
    train_data = dataset['train']
    df = pd.DataFrame(train_data)
    return df


def sample_and_save_data(df: pd.DataFrame, sample_size: int, save_path: str) -> None:
    """
    Sample data from a DataFrame and save it to a CSV file.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - sample_size (int): Size of the sample.
    - save_path (str): Path to save the sampled data.
    """
    sampled_df = df.sample(n=sample_size)
    sampled_df.to_csv(save_path)


def main() -> None:
    """
    Main function to execute the data loading, sampling, and saving process.
    """
    # Load and process data
    df = load_and_process_data()

    # Sample and save data
    sample_size = 150000
    save_path = '../data/chatgpt_paraphrases_modified.csv'
    sample_and_save_data(df, sample_size, save_path)


if __name__ == "__main__":
    main()
