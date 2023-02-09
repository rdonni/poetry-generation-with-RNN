#!/usr/bin/env python3

import tensorflow as tf

from build_model import TextGenerationModel
from config import DatasetCreationConfig, ModelConfig, TrainConfig
from dataset_creation import build_training_dataset


def train(dataset_config: DatasetCreationConfig,
          model_config: ModelConfig,
          train_config: TrainConfig):
    input_sequence, label, total_nb_words = build_training_dataset(dataset_config.poems_dataset_path,
                                                                   dataset_config.sequence_size,
                                                                   dataset_config.char_level_tokenizing,
                                                                   dataset_config.tokenizer_saving_path)
    print('Dataset Created')
    model = TextGenerationModel(total_nb_words, embedding_dim=model_config.embedding_dim)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.build((None, 1))
    model.summary()

    model.fit(input_sequence,
              label,
              epochs=train_config.nb_epochs,
              batch_size=train_config.batch_size,
              verbose=1,
              callbacks=[tf.keras.callbacks.TensorBoard(train_config.logdir, histogram_freq=1),
                         tf.keras.callbacks.ModelCheckpoint(
                             filepath=train_config.model_output_path,
                             save_best_only=True,
                             monitor='loss',
                             mode='min',
                             save_freq='epoch',
                             verbose=1)
                         ])


dataset_config = DatasetCreationConfig(
    poems_dataset_path='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/PoetryFoundationData.csv',
    sequence_size=32,
    char_level_tokenizing=False,
    tokenizer_saving_path='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/tokenizer.json'
)
model_config = ModelConfig(
    embedding_dim=8
)
train_config = TrainConfig(
    nb_epochs=50,
    batch_size=32,
    model_output_path='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/model',
    logdir='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/logs'
)

train(dataset_config, model_config, train_config)
