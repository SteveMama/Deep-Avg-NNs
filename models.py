# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class DAN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, embeddings):
        """

        Args:
            input_dim:
            hidden_dim:
            output_dim:
            embeddings:
        """
        super(DAN,self).__init__()

        self.embedding = embeddings.get_initialized_embedding_layer(frozen= False)
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(0.35)
        self.activation = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, word_indices):

        embeddings_layer = self.embedding(word_indices)
        averaged_embeddings_layer = torch.mean(embeddings_layer, dim=0)

        hidden = self.activation(self.hidden_layer(averaged_embeddings_layer))
        hidden = self.dropout_layer(hidden)

        output = self.output_layer(hidden)

        return self.softmax(output)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model, word_embeddings):
        #raise NotImplementedError
        self.model = model
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str]) -> int:

        words_embeddings


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """




    raise NotImplementedError

