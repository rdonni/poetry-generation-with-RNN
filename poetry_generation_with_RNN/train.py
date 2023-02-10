#!/usr/bin/env python3
from typing import List

import tensorflow as tf
import numpy as np
from poetry_generation_with_RNN.build_model import TextGenerationModel
from poetry_generation_with_RNN.config import ModelConfig, TrainConfig
import wandb
from wandb.keras import WandbCallback

wandb.init(project="poetry-generation-with-RNN")
wandb.run.name = 'test-run'


def build_model_and_train(x: List[int],
                          y: List[int],
                          num_words_in_corpus: int,
                          model_config: ModelConfig,
                          train_config: TrainConfig):

    model = TextGenerationModel(total_num_words_in_corpus=num_words_in_corpus, embedding_dim=model_config.embedding_dim)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=train_config.learning_rate),
                  metrics=['accuracy'])
    model(np.array([x[0]]))
    model.summary()

    model.fit(x,
              y,
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
                             verbose=2),
                         WandbCallback()
                         ])
