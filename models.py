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

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SentimentExample], word_embeddings: WordEmbeddings):
        self.examples = examples
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_dim = word_embeddings.get_embedding_length()

        if not self.word_indexer.contains("UNK"):
            unk_vector = np.zeros(self.embedding_dim)
            self.word_indexer.add_and_get_index("UNK")
            self.word_embeddings.add_embedding("UNK", unk_vector)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex.words
        label = ex.label

        embeddings = []
        for word in words:
            embedding = self.word_embeddings.get_embedding(word)
            if embedding is None:
                embedding = self.word_embeddings.get_embedding("UNK")
            embeddings.append(embedding)
        embeddings = torch.tensor(embeddings, dtype=torch.float)

        avg_embedding = embeddings.mean(dim=0)

        return avg_embedding, torch.tensor(label, dtype=torch.long)

class DeepAveragingNetwork(nn.Module):
    def __init__(self, embedding_dim, args):
        super(DeepAveragingNetwork, self).__init__()

        hidden_size = args.hidden_size


        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, avg_embedding):
        """
        Forward pass of the network.

        :param avg_embedding: Input tensor of shape (batch_size, embedding_dim).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        out = self.fc1(avg_embedding)
        out = self.relu(out)
        logits = self.fc2(out)

        return logits

class DANClassifier(SentimentClassifier):
    def __init__(self, model, word_embeddings):
        self.model = model
        self.model.eval()
        self.word_embeddings = word_embeddings
        self.word_indexer = word_embeddings.word_indexer
        self.embedding_dim = word_embeddings.get_embedding_length()

    def predict(self, words: List[str]) -> int:
        """
        Predicts the sentiment label for a single sentence.

        :param words: List of words in the sentence.
        :return: Predicted label (0 or 1).
        """
        embeddings = []
        for word in words:
            embedding = self.word_embeddings.get_embedding(word)
            if embedding is None:
                embedding = self.word_embeddings.get_embedding("UNK")
            embeddings.append(embedding)
        embeddings = torch.tensor(embeddings, dtype=torch.float)

        avg_embedding = embeddings.mean(dim=0).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(avg_embedding)
            probabilities = nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        return prediction

def train_deep_averaging_network(args, train_exs: List[SentimentExample],
                                 dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> SentimentClassifier:
    """
    Trains a Deep Averaging Network (DAN) for sentiment analysis using DataLoader.

    :param args: Command-line arguments.
    :param train_exs: Training examples.
    :param dev_exs: Development examples.
    :param word_embeddings: Pre-trained word embeddings.
    :return: A trained SentimentClassifier model.
    """
    train_dataset = SentimentDataset(train_exs, word_embeddings)
    dev_dataset = SentimentDataset(dev_exs, word_embeddings)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    embedding_dim = word_embeddings.get_embedding_length()
    model = DeepAveragingNetwork(embedding_dim, args)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        total_loss = 0.0

        for avg_embeddings, labels in train_loader:

            optimizer.zero_grad()

            logits = model(avg_embeddings)
            target = labels

            loss = criterion(logits, target)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for avg_embeddings, labels in dev_loader:
                logits = model(avg_embeddings)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Dev Accuracy: {accuracy:.4f}")
        model.train()
    return DANClassifier(model, word_embeddings)

