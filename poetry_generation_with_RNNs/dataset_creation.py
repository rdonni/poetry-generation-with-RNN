import pandas as pd
import tensorflow as tf
import numpy as np
from typing import List


def build_training_dataset(
        poem_dataset_path: str,
        sequence_size: int,
        char_level_tokenizing: bool,
        tokenizer_saving_path: str
):
    df = pd.read_csv(poem_dataset_path, nrows=500)
    poems_array = poems_df_to_poems_array(df)
    preprocessed_sequences = split_poems_into_sequences(poems_array)

    tokenizer = build_tokenizer(preprocessed_sequences, tokenizer_saving_path, char_level_tokenizing)
    total_nb_words = len(tokenizer.word_index) + 1

    input_sequence, label = sequences_to_training_data(preprocessed_sequences, tokenizer, sequence_size)
    return input_sequence, label, total_nb_words


def poems_df_to_poems_array(poems_df: pd.DataFrame) -> List[str]:
    text_df = poems_df['Poem'].replace('\r', '')
    poems_array = text_df.loc[1:].dropna().to_numpy()
    return poems_array


def split_poem_into_verses(text: str):
    return text.lower().replace('\r', '').split("\n")


def split_poems_into_sequences(poems_array: List[str]):
    preprocessed_verses = [verse for p in poems_array for verse in split_poem_into_verses(p) if verse not in ['', ' ']]
    return preprocessed_verses


def sequences_to_training_data(text_sequences, tokenizer, padded_sequence_size):
    input_sequences, labels = [], []

    for verse in text_sequences:
        token_list = tokenizer.texts_to_sequences([verse])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i])
            labels.append(token_list[i])

    padded_inputs = pad_sequences(input_sequences, padded_sequence_size)
    categorical_labels = labels_to_categorical(labels, tokenizer)

    return np.array(padded_inputs), np.array(categorical_labels)


def labels_to_categorical(tokenized_labels, tokenizer):
    total_words = len(tokenizer.word_index) + 1
    return tf.keras.utils.to_categorical(tokenized_labels, num_classes=total_words)


def pad_sequences(tokenized_inputs, padded_sequence_size):
    input_sequences = tf.keras.utils.pad_sequences(tokenized_inputs, maxlen=padded_sequence_size, padding='pre')
    return input_sequences


def build_tokenizer(text_sequences, tokenizer_saving_path, char_level):
    if char_level:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True,
                                                          filters='')
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(text_sequences)

    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_saving_path, "w") as outfile:
        outfile.write(tokenizer_json)
    return tokenizer
