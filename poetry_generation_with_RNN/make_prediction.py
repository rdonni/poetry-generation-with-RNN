#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import json

from dataset_creation import pad_sequences


def generate(model_path, tokenizer_path, prompt_text, nb_words):
    with open(tokenizer_path) as jsonfile:
        tokenizer_json = json.load(jsonfile)
    json_str = json.dumps(tokenizer_json)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_str)
    model = tf.keras.models.load_model(model_path)

    for _ in range(nb_words):
        tokenized_prompt = tokenizer.texts_to_sequences([prompt_text])[0]
        padded_prompt = pad_sequences([tokenized_prompt], 32)
        model_pred = model(padded_prompt)
        id_pred = np.argmax(model_pred, axis=1)
        pred = tokenizer.sequences_to_texts([id_pred])[0]
        prompt_text = prompt_text + f" {pred} "

    return prompt_text


model_path_ = '/Users/rayanedonni/Documents/Projets_persos/poetry_creation/model'
tokenizer_path_ = '/Users/rayanedonni/Documents/Projets_persos/poetry_creation/tokenizer.json'
prompt_text_ = 'The little boy'
nb_words_ = 20

generated_text = generate(model_path_, tokenizer_path_, prompt_text_, nb_words_)
print(generated_text)
