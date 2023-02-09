from dataclasses import dataclass


@dataclass
class DatasetCreationConfig:
    poems_dataset_path: str
    sequence_size: int
    char_level_tokenizing: bool
    tokenizer_saving_path: str


@dataclass
class ModelConfig:
    embedding_dim: int


@dataclass
class TrainConfig:
    nb_epochs: int
    batch_size: int
    model_output_path: str
    logdir: str
