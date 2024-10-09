# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from typing import List
# from sentiment_data import *
# import numpy as np
#
# class SentimentClassifier(object):
#     """
#     Base class for sentiment classifiers.
#     """
#     def predict(self, words: List[str]) -> int:
#         """
#         Predicts the sentiment label for a given sentence.
#
#         :param words: List of words in the sentence.
#         :return: 1 if positive sentiment, 0 if negative sentiment.
#         """
#         raise Exception("Don't call me, call my subclasses")
#
#     def predict_all(self, sentences: List[List[str]]) -> List[int]:
#         """
#         Predicts sentiment labels for a list of sentences.
#
#         :param sentences: List of sentences, each a list of words.
#         :return: List of predictions.
#         """
#         return [self.predict(words) for words in sentences]
#
# class TrivialSentimentClassifier(SentimentClassifier):
#     """
#     A trivial classifier that always predicts positive sentiment.
#     """
#     def predict(self, words: List[str]) -> int:
#         return 1
#
#
# class customDataset(torch.utils.data.Dataset):
#     def __init__(self, examples: List[SentimentExample], word_embeddings : WordEmbeddings):
#         self.examples = examples
#         self.word_embeddings = word_embeddings
#         self.word_indexer = word_embeddings.word_indexer
#         self.embedding_dim = word_embeddings.get_embedding_length()
#
#         if not self.word_indexer.contains("UNK"):
#             unk_vector = np.zeroes(self.embedding_dim)
#             self.word_indexer.add_and_get_index("UNK")
#             self.word_embeddings.add_embedding("UNK", unk_vector)
#
#
#     def __len__(self):
#         return len(self.examples)
#
#     def __getitem__(self, item_index):
#         ex = self.examples[item_index]
#         words = ex.words
#         label = ex.label
#
#         embeddings = []
#         for word in words:
#             embedding = self.word_embeddings.get_embedding(word)
#             if embedding is None:
#                 embedding = self.word_embeddings.get_embedding("UNK")
#             embeddings.append(embedding)
#         embeddings = torch.tensor(embeddings, dtype=torch.float)
#
#         avg_embedding = embeddings.mean(dim=0)
#
#         return avg_embedding, torch.tensor(label, dtype=torch.long)
#
#
# class DAN(nn.Module):
#     def __init__(self, embedding_dim, args):
#         super(DAN, self).__init__()
#
#         hidden_size = args.hidden_size #hidden layer size
#         self.fc1 = nn.Linear(embedding_dim, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, 2) #binary classification
#
#
#     def forward(self, avg_embedding):
#
#         output = self.fc1(avg_embedding)
#         output = self.relu(output)
#         logits = self.fc2(output)
#
#         return logits
#
# class DANClassifier(SentimentClassifier):
#
#     def __init__(self, model, word_embeddings):
#         self.model = model
#         self.model.eval()
#         self.word_embeddings = word_embeddings
#         self.word_indexer = word_embeddings.word_indexer
#         self.embedding_dim = word_embeddings.get_embedding_length()
#
#     def predict(self, words: List[str]) -> int:
#
#         embeddings = []
#         for word in words:
#             embedding = self.word_embeddings.get_embedding(word)
#             if embedding is None:
#                 embedding = self.word_embeddings.get_embedding("UNK")
#             embeddings.append(embedding)
#         embeddings = torch.tensor(embeddings, dtype=torch.float)
#
#         avg_embeddings = embeddings.mean(dim=0).unsqueeze(0)
#
#         with torch.no_grad():
#             logits = self.model(avg_embeddings)
#             cls_prob = F.softmax(logits, dim=1)
#             pred = torch.argmax(cls_prob, dim=1).item()
#
#         return pred
#
#
# def train_deep_averaging_network(args, train_exs: List[SentimentExample],
#                                  dev_exs: List[SentimentExample],
#                                  word_embeddings: WordEmbeddings) -> SentimentClassifier:
#
#     train_dataset = customDataset(train_exs, word_embeddings)
#     dev_dataset = customDataset(dev_exs, word_embeddings)
#
#     train_dataset = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle = True
#     )
#
#     dev_dataset = torch.utils.data.DataLoader(
#         dev_dataset, batch_size = args.batch_size, shuffle = False
#     )
#
#     embedding_dim = word_embeddings.get_embedding_length()
#     model = DAN(embedding_dim, args)
#     model.train()
#
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr = args.lr)
#
#     num_epochs = args.num_epochs
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#
#         for avg_embeddings, labels in train_dataset:
#             optimizer.zero_grad()
#             logits = model(avg_embeddings)
#             loss = criterion(logits, labels)
#             total_loss += loss.item()
#             loss.backward()
#             optimizer.step()
#
#         avg_loss = total_loss/ len(train_dataset)
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
#
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for avg_embeddings, labels in dev_dataset:
#                 logits = model(avg_embeddings)
#                 predictions = torch.argmax(logits, dim=1)
#                 correct += (predictions == labels).sum().item()
#                 total += labels.size(0)
#         accuracy = correct / total
#         print(f"Dev Accuracy: {accuracy:.4f}")
#         model.train()
#
#     return DANClassifier(model, word_embeddings)
#

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
    if args.random_init:
        vocab_size = len(word_embeddings.word_indexer)
        embedding_dim = args.embedding_dim
        model = DeepAveragingNetworkRandom(vocab_size, embedding_dim, args)
    else:
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


        permutation = torch.randperm(len(train_indices))

        for i in range(0, len(train_indices), batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

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

        embeds = self.embeddings(x)
        mask = (x != 0).unsqueeze(-1).float()
        embeds = embeds * mask
        sum_embeds = torch.sum(embeds, dim=1)
        lengths = torch.sum(mask, dim=1)
        lengths[lengths == 0] = 1
        avg_embeds = sum_embeds / lengths  # Shape: (batch_size, embedding_dim)
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class DeepAveragingNetworkRandom(nn.Module):
    def __init__(self, vocab_size, embedding_dim, args):
        super(DeepAveragingNetworkRandom, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        hidden_size = args.hidden_size


        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)  # Output layer for binary classification

    def forward(self, x):

        embeds = self.embeddings(x)

        mask = (x != 0).unsqueeze(-1).float()  # Shape: (batch_size, seq_length, 1)
        embeds = embeds * mask

        sum_embeds = torch.sum(embeds, dim=1)  # Shape: (batch_size, embedding_dim)

        lengths = torch.sum(mask, dim=1)  # Shape: (batch_size, 1)

        lengths[lengths == 0] = 1

        avg_embeds = sum_embeds / lengths  # Shape: (batch_size, embedding_dim)
        out = self.fc1(avg_embeds)
        out = self.relu(out)
        out = self.fc2(out)

        return out


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
