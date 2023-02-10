import tensorflow as tf


class TextGenerationModel(tf.keras.Model):
    def __init__(self, total_num_words_in_corpus, embedding_dim: int):
        super().__init__()
        self.total_num_words_in_corpus = total_num_words_in_corpus
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(self.total_num_words_in_corpus, embedding_dim)
        self.gru_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, activation='relu'))
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True, activation='relu'))
        self.dropout_2 = tf.keras.layers.Dropout(0.2)
        self.gru_3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, activation='relu'))
        self.dropout_3 = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(self.total_num_words_in_corpus, activation='softmax')

    def call(self, x):
        embedded_input = self.embedding(x)
        x = self.gru_1(embedded_input)
        x = self.dropout_1(x)
        x = self.gru_2(x)
        x = self.dropout_2(x)
        x = self.gru_3(x)
        x = self.dropout_3(x)
        output = self.dense(x)
        return output


