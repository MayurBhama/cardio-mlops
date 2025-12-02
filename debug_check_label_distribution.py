# debug_check_label_distribution.py
import pandas as pd

def show_distribution(name, path, **read_kwargs):
    df = pd.read_csv(path, **read_kwargs)
    dist = df["cardio"].value_counts(normalize=True).to_dict()
    print(f"{name}  ->  {dist}")

if __name__ == "__main__":
    show_distribution("RAW", "data/raw/cardio_train.csv", sep=";")
    show_distribution("CLEANED", "data/processed/cardio_cleaned.csv")
    show_distribution("FEATURED", "data/processed/cardio_featured.csv")
