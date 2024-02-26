"""Taller evaluable presencial"""

import nltk
import pandas as pd


def load_data(input_file):

    data = pd.read_csv(input_file, sep="\t")
    return data

def create_key(df, n):
    df = df.copy()
    df["key"] = df["text"]
    df["key"] = (
        df["key"]
        .str.strip()
        .str.lower()
        .str.replace("-","")
        .str.translate(
            str.maketrans("", "", "!\"#$%&'()++,-./:;<=<?@[\\]^_")
        )
        .str.replace(".","")
        .str.split()
        .str.join("")
        .apply(lambda x: [x[i:i + n] for i in range(len(x)- n + 1)])
        .apply(lambda x: sorted(set(x)))
        .str.join(" ")
    )
    return df


def generate_cleaned_column(df):
    df = df.copy()
    keys = df.sort_values(by = ["key", "text"]).copy()
    keys = df.groupby("key").first().reset_index()
    keys = keys.set_index("key")["text"].to_dict()
    df["cleaned"] = df["key"].map(keys)
    return df


def save_data(df, output_file):
    """Guarda el DataFrame en un archivo"""

    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns={"cleaned": "text"})
    df.to_csv(output_file, index=False)


def main(input_file, output_file, n=2):
    """Ejecuta la limpieza de datos"""

    df = load_data(input_file)
    df = create_key(df, n)
    df = generate_cleaned_column(df)
    df.to_csv("test.csv", index=False)
    save_data(df, output_file)


if __name__ == "__main__":
    main(
        input_file="input.txt",
        output_file="output.txt",
    )
