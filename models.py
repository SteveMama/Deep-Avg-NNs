# models.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from sentiment_data import *
import numpy as np


class SentimentClassifier(object):
    """
    Base class for sentiment classifiers.
    """
    def predict(self, words: List[str]) -> int:
        """
        Predicts the sentiment label for a given sentence.

        :param words: List of words in the sentence.
        :return: 1 if positive sentiment, 0 if negative sentiment.
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, sentences: List[List[str]]) -> List[int]:
        """
        Predicts sentiment labels for a list of sentences.

        :param sentences: List of sentences, each a list of words.
        :return: List of predictions.
        """
        return [self.predict(words) for words in sentences]


class TrivialSentimentClassifier(SentimentClassifier):
    """
    A trivial classifier that always predicts positive sentiment.
    """
    def predict(self, words: List[str]) -> int:
        return 1


def train_deep_averaging_network(args, train_exs: List[SentimentExample],
                                 dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> SentimentClassifier:
    """
    Trains a Deep Averaging Network (DAN) for sentiment analysis.

    :param args: Command-line arguments.
    :param train_exs: Training examples.
    :param dev_exs: Development examples.
    :param word_embeddings: Pre-trained word embeddings.
    :return: A trained SentimentClassifier model.
    """
    # Initialize the model
    model = DeepAveragingNetwork(word_embeddings, args)
    model.train()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare data for batching
    word_indexer = word_embeddings.word_indexer

    # Convert training examples to word indices and labels
    train_sentences = [ex.words for ex in train_exs]
    train_labels = [ex.label for ex in train_exs]

    # Map words to indices, handling unknown words
    train_indices = [
        torch.tensor([word_indexer.index_of(word) if word_indexer.contains(word)
                      else word_indexer.index_of("UNK") for word in sentence], dtype=torch.long)
        for sentence in train_sentences
    ]

    train_labels = torch.tensor(train_labels, dtype=torch.long)

    # Batch size
    batch_size = args.batch_size

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Shuffle the data at the beginning of each epoch
        permutation = torch.randperm(len(train_indices))

        for i in range(0, len(train_indices), batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_sentences = [train_indices[idx] for idx in indices]
            batch_labels = train_labels[indices]

            # Pad sequences to the same length
            batch_sentences_padded = nn.utils.rnn.pad_sequence(
                batch_sentences, batch_first=True, padding_value=0
            )  # Shape: (batch_size, max_seq_length)

            # Forward pass
            outputs = model(batch_sentences_padded)  # Shape: (batch_size, num_classes)

            # Compute loss
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * len(batch_sentences)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_indices)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Return the trained model wrapped in a classifier
    return DANClassifier(model, word_indexer)


class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, args):
        super(DeepAveragingNetwork, self).__init__()
        embedding_dim = word_embeddings.get_embedding_length()
        self.embeddings = word_embeddings.get_initialized_embedding_layer(frozen=False)
        self.embeddings.padding_idx = 0  # Ensure padding index is ignored

        hidden_size = args.hidden_size

        # Define the feedforward network
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Output layer for binary classification

    def forward(self, x):
        """
        Forward pass of the network.

        :param x: Input tensor of shape (batch_size, sequence_length).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        # Obtain embeddings: (batch_size, seq_length, embedding_dim)
        embeds = self.embeddings(x)

        # Create a mask to ignore padding tokens
        mask = (x != 0).unsqueeze(-1).float()  # Shape: (batch_size, seq_length, 1)

        # Zero out embeddings corresponding to padding tokens
        embeds = embeds * mask

        # Sum the embeddings
        sum_embeds = torch.sum(embeds, dim=1)  # Shape: (batch_size, embedding_dim)

        # Compute the lengths of each sequence (number of non-padding tokens)
        lengths = torch.sum(mask, dim=1)  # Shape: (batch_size, 1)

        # Avoid division by zero
        lengths[lengths == 0] = 1

        # Compute the average embeddings
        avg_embeds = sum_embeds / lengths  # Shape: (batch_size, embedding_dim)

        # Pass through the feedforward network
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)  # Output logits (before softmax)

        return out  # Raw scores for each class


class DANClassifier(SentimentClassifier):
    def __init__(self, model, word_indexer):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.word_indexer = word_indexer

    def predict(self, words: List[str]) -> int:
        """
        Predicts the sentiment label for a single sentence.

        :param words: List of words in the sentence.
        :return: Predicted label (0 or 1).
        """
        # Convert words to indices
        indices = [self.word_indexer.index_of(word) if self.word_indexer.contains(word)
                   else self.word_indexer.index_of("UNK") for word in words]

        # Convert to tensor and add batch dimension
        indices_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # Shape: (1, seq_length)

        with torch.no_grad():
            outputs = self.model(indices_tensor)  # Shape: (1, num_classes)
            probabilities = nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        return prediction

    def predict_all(self, sentences: List[List[str]]) -> List[int]:
        """
        Predicts sentiment labels for a list of sentences.

        :param sentences: List of sentences, each a list of words.
        :return: List of predicted labels.
        """
        self.model.eval()
        predictions = []

        # Batch prediction for efficiency
        batch_size = 64  # Adjust as needed
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]

            # Convert words to indices
            batch_indices = [
                torch.tensor([self.word_indexer.index_of(word) if self.word_indexer.contains(word)
                              else self.word_indexer.index_of("UNK") for word in sentence], dtype=torch.long)
                for sentence in batch_sentences
            ]

            # Pad sequences
            batch_sentences_padded = nn.utils.rnn.pad_sequence(
                batch_indices, batch_first=True, padding_value=0
            )

            with torch.no_grad():
                outputs = self.model(batch_sentences_padded)
                probabilities = nn.functional.softmax(outputs, dim=1)
                batch_predictions = torch.argmax(probabilities, dim=1).tolist()
                predictions.extend(batch_predictions)

        return predictions
