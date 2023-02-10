from dataclasses import dataclass


@dataclass
class DatasetCreationConfig:
    poems_dataset_path: str
    num_poems: int
    sequence_size: int
    char_level_tokenizing: bool
    tokenizer_output_path: str


@dataclass
class ModelConfig:
    embedding_dim: int


@dataclass
class TrainConfig:
    nb_epochs: int
    batch_size: int
    learning_rate: float
    model_output_path: str
    logdir: str
