from typing import List

from poetry_generation_with_RNN.config import DatasetCreationConfig, ModelConfig, TrainConfig
from poetry_generation_with_RNN.dataset_creation import build_training_dataset
from poetry_generation_with_RNN.train import build_model_and_train
from poetry_generation_with_RNN.make_prediction import generate

import wandb

wandb.init(project="poetry-generation-with-RNN")
wandb.run.name = 'test-run'


def train_and_predict(dataset_config: DatasetCreationConfig,
                      model_config: ModelConfig,
                      train_config: TrainConfig,
                      test_sequences: List[str]):

    input_sequence, label, total_nb_words = build_training_dataset(dataset_config.poems_dataset_path,
                                                                   dataset_config.num_poems,
                                                                   dataset_config.sequence_size,
                                                                   dataset_config.char_level_tokenizing,
                                                                   dataset_config.tokenizer_output_path)

    build_model_and_train(x=input_sequence,
                          y=label,
                          num_words_in_corpus=total_nb_words,
                          model_config=model_config,
                          train_config=train_config)

    for text in test_sequences:
        generate(model_path=train_config.model_output_path,
                 tokenizer_path=dataset_config.tokenizer_output_path,
                 prompt_text=text,
                 nb_words=30)


if __name__ == '__main__':
    dataset_config_ = DatasetCreationConfig(
        poems_dataset_path='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/PoetryFoundationData.csv',
        num_poems=10,
        sequence_size=32,
        char_level_tokenizing=False,
        tokenizer_output_path='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/tokenizer.json'
    )
    model_config_ = ModelConfig(
        embedding_dim=8
    )
    train_config_ = TrainConfig(
        nb_epochs=10,
        learning_rate=0.001,
        batch_size=128,
        model_output_path='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/model',
        logdir='/Users/rayanedonni/Documents/Projets_persos/poetry_creation/logs'
    )

    test_sentences_ = [
        'The little boy',
        'A fall day'
    ]

    wandb.config = {
        "num_poems_in_dataset": dataset_config_.num_poems,
        "char_level_tokenizing": dataset_config_.char_level_tokenizing,
        "sequence_size": dataset_config_.sequence_size,
        "embedding_dim": model_config_.embedding_dim,
        "learning_rate": train_config_.learning_rate,
        "epochs": train_config_.nb_epochs,
        "batch_size": train_config_.batch_size,
    }

    train_and_predict(dataset_config_, model_config_, train_config_, test_sentences_)
