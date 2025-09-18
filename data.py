"""
Data Acquisition and Exploration Module
Why: We need labeled datasets to train and test our fake news detection model.
How: Load datasets from Hugging Face, explore the data structure, and prepare for preprocessing.
"""

import pandas as pd
from datasets import load_dataset

def load_fake_news_dataset(dataset_name="mazihan880/AMG", split="train"):
    """
    Load a multimodal fake news dataset from Hugging Face.
    Why: AMG dataset contains text, images, videos, and labels for fake/real news.
    How: Use datasets library to download and load the data.
    """
    try:
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(dataset[split])
        print(f"Loaded {len(df)} samples from {dataset_name}")
        print("Columns:", df.columns.tolist())
        print("Sample data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_data(df):
    """
    Explore the dataset to understand its structure.
    Why: Understanding data helps in preprocessing and feature extraction.
    How: Print statistics, label distribution, and sample entries.
    """
    if df is None:
        return

    print("\n--- Data Exploration ---")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check for label distribution
    if 'label' in df.columns:
        print("\nLabel distribution:")
        print(df['label'].value_counts())

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Sample text
    if 'text' in df.columns:
        print("\nSample text:")
        print(df['text'].iloc[0][:200] + "...")

if __name__ == "__main__":
    # Load and explore the AMG dataset
    df = load_fake_news_dataset()
    explore_data(df)