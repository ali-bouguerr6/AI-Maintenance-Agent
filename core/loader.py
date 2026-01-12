import pandas as pd

def load_dataset(path="data/ai4i.csv"):
    df = pd.read_csv(path)
    return df
